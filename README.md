# VaaniEdge — On-Device Speech AI Voice Agent

**IISc Computational Intelligence Lab** · Android · Snapdragon 8+ Gen 1 (OnePlus 11R 5G, `arm64-v8a`)

A Hinglish + English voice agent whose speech stack runs entirely on the phone —
Silero VAD → KV-cached Whisper STT with step-0 language ID → Hinglish post-correction →
streaming cloud LLM with a Groq ⇄ Gemini failover cascade → clause-level Indic VITS TTS —
built from first principles on the official **ONNX Runtime** (Microsoft, MIT). It answers the
four IISc evaluation scenarios (Bengaluru weather tomorrow, a 5 PM meeting reminder, a 7 AM alarm
in Hinglish, today's schedule) with grounded local tools, and shows a live latency HUD for every
turn. Every engine, DSP routine, codec and UI component under `in.iisc.vaani.*` is original work;
models, runtime and fonts are attributed dependencies (see [`NOTICE`](NOTICE)).

> `./gradlew testDebugUnitTest assembleDebug assembleRelease`
> → **BUILD SUCCESSFUL, 90 unit tests green**, a debug APK and a signed release APK. Every
> shipped graph loads and runs under onnxruntime **1.18.0** (the APK's runtime) on the
> host. Speech quality and latency below are **measured on the build host**; the phone's own
> numbers appear in the in-app HUD and in logcat (`VaaniTelemetry`).

---

## Architecture

```
 mic ─▶ AudioRecord(VOICE_RECOGNITION,16k) ─▶ NEON RMS (waveform) ─▶ Silero VAD (64-sample context)
          │ 2 speech chunks ──────────────▶ onSpeechStart() ─▶ async HEAD pre-warm (TCP+TLS1.3+HTTP/2)
          │ 500 ms trailing silence ──────▶ t_end  (E2E clock starts)
          ▼
   NEON log-Mel [1,80,3000] ─▶ Whisper encoder (INT8) ─▶ decoder([SOT]) ─▶ LID: <|en|> vs <|hi|>
          (C++, zero-frame skip)                     │  cross-attn K/V computed once, reused by reference
                                                     ▼
                         decoder_with_past (KV cache, fused INT8 logits) ─▶ masked greedy ─▶ raw text
          ▼
   HinglishCodec.displayTranscript: phonetic lexicon + compound splitter + context rules (< 0.5 ms)
          ▼
   LLM SSE stream — automatic failover before the first token:
     Groq Qwen3.8-27B ─▶ Groq GPT-OSS-20B ─▶ Gemini 3.5 Flash-Lite ─▶ Gemini 3.5 Flash ─▶ on-device LocalResponder
          │ tokens ─▶ ResponseStreamProcessor: [ACTION:...] tags → tools, clauses → speech
          ▼
   clause ─▶ HinglishCodec.toAcousticDevanagari(lang) ─▶ IndicPhonemizer (espeak-compatible IPA)
          ─▶ VITS (fp16-stored) ─▶ PCM queue ─▶ low-latency AudioTrack   (first sample = t_audio)
                                   (clause N+1 synthesises while N plays; tap = barge-in)
```

| Package | What it does |
|---|---|
| `dsp` + `cpp/vaani_dsp_neon.cpp` | NEON DSP: Whisper-exact log-Mel (planned 2⁴·25 FFT, all-zero padding frames skipped exactly, sparse Mel rows), Kaiser-sinc resampler (22.05→16 kHz), RMS, float→PCM16. Also exports a C ABI so the host lab tests this exact file. |
| `neural` | `VaaniSttEngine` (KV cache, step-0 2-way LID + manual override, Whisper non-speech/special/script masks, suppress_blank, repetition guard, zero-copy direct buffers), `VaaniTtsEngine` (Piper sentence layout, peak-normalised), `VaaniVadEngine`. Sessions: `ALL_OPT`, 4 intra-op threads, `session.intra_op.allow_spinning=1`. |
| `codec` | `IndicPhonemizer`: Devanagari → espeak-ng-compatible IPA (aspiration `ʰ`, palatal `c ɟ`, schwa deletion, Hindi stress, nasals, gemination), Piper ids. `HinglishCodec`: language-aware numbers/times (`साढ़े चार` vs `four thirty`), curated Indian-English + Hinglish lexicons, CMUdict → Devanagari, acronyms; and the **STT post-corrector** (`displayTranscript`): Hinglish phonetic key + curated variants, dynamic-programming compound splitter, context rules, ASR stutter cleanup. |
| `llm` | `LlmStreamingClient` (OkHttp HTTP/2, pre-warm, Groq + Gemini SSE, **cross-provider failover cascade** with per-model cool-downs, 4 s first-token timeout, per-call connection-reuse evidence), `ResponseStreamProcessor` (clause cutting, tag parsing), `PromptBuilder`, `LlmSettings`, `LlmModels`. |
| `tools` | `LocalToolManager`: Open-Meteo forecast cached at launch (0 ms lookup, offline fallback), reminders → Calendar intent, alarms → `AlarmClock.ACTION_SET_ALARM`, IISc lab schedule anchored to "now"; `LocalResponder` answers all four scenarios offline. |
| `agent`, `audio`, `runtime`, `ui` | Turn orchestration + telemetry, capture/playback pipeline, asset→engine glue with warm-up, Jetpack Compose Material 3 UI. |

### UI
Telemetry HUD (`STT (Local INT8)`, `LLM TTFT`, `TTS First-Audio`, `Total E2E Latency` with a
proportional STT | LLM | TTS latency bar), status line, language (`Auto EN/HI` / `English` /
`Hinglish`) and LLM chips, a strip with the four IISc prompts (**Speak** live or **Loopback**:
the phone synthesises the prompt, feeds that audio to STT → LLM → TTS and logs the turn), the
conversation with `[EN]`/`[HI-EN]` badges, STT ms, WER vs. the prompt, the corrected transcript
with the `raw ASR` line under it, the answering model (`… (failover)` when a rung was skipped),
replay and a Devanagari/IPA phonetics view, action cards (weather, alarm, reminder, schedule), a
60 fps waveform (saffron = your voice, phosphor = the agent) and the mic button
(`Tap to Speak (Auto-VAD)` → `Listening...` → `Transcribing...` → `Streaming...` →
`Speaking (Tap to Barge-In)`). Settings: primary model with the live failover order, model-id
override, Groq/Gemini keys, cloud on/off, recognition language.

---

## LLM: Groq ⇄ Gemini failover cascade

| Rung | Model (request settings) | TTFT* | `[ACTION:…]` tag | Reply | Free tier (observed) |
|---|---|---|---|---|---|
| 1 (default) | Groq `qwen/qwen3.8-27b` (`reasoning_effort: none`) | **111 ms** | ✔ line 1 | Roman Hinglish | 1,000 req/day · 8k tokens/min |
| 2 | Groq `openai/gpt-oss-20b` (`low`, reasoning hidden) | 472 ms | ✔ line 1 | English | 1,000 req/day · 8k tokens/min |
| 3 | Gemini `gemini-3.5-flash-lite` (`thinkingLevel: minimal`) | 1,137 ms | ✔ line 1 | Roman Hinglish | — |
| 4 | Gemini `gemini-3.5-flash` (`thinkingBudget: 0`) | HTTP 429 | — | — | **20 req/day** (daily cap) |
| 5 | on-device `LocalResponder` | no network | ✔ (grounded actions) | EN / Hinglish | unlimited |
| manual | Groq `openai/gpt-oss-120b` (`low`, hidden) | 667 ms | ✔ (same line as reply; parsed anyway) | Roman Hinglish | 1,000 req/day |

\* One streamed request per model from the build host (`speech_lab/llm_smoke_once.py`, quota-safe):
the app's real system prompt + the Hinglish alarm turn, on a pre-warmed connection, request sent →
first content token. n = 1 per model, so read these as indicative, not as medians.

- Groq has retired **`llama-3.3-70b-versatile`** and **`llama-3.1-8b-instant`** (HTTP 404
  `model_not_found`, 18 Oct 2026); Qwen3.8-27B is now the default primary.
- **Failover** happens only before the first token, so a reply is never restarted mid-sentence.
  Triggers: any HTTP error (429/404/5xx/401…), a network error, no token within 4 s, or an empty
  reply. A network error also skips that provider's remaining rungs for the turn. Rungs whose
  provider has no key are skipped.
- **Cool-downs** stop later turns from re-trying a failed rung: 429 honours `Retry-After` (Groq) or
  `RetryInfo.retryDelay` (Gemini); 400/401/403/404 sit out 30 min; 5xx/timeouts 20 s. Saving
  Settings resets them.
- The Iteration-2 Gemini 429s were the **daily** free-tier cap on 3.5 Flash (`GenerateRequestsPerDay…
  FreeTier`, 20 requests), not a per-minute window.
- Telemetry: the reply bubble names the model that answered; logcat `VaaniTelemetry` adds
  `llm_model_id` and `failovers`.

---

## Measured results (host: Xeon 8259CL, 4 threads; `speech_lab/evaluation_metrics.json`)

| Closed loop: Kotlin G2P → VITS → C++ Mel → Whisper | Iteration 1 | Iteration 2 | Iteration 3 |
|---|---|---|---|
| Phoneme coverage of `phoneme_id_map` | 99.76 % (breathy `ʱ` dropped) | **100 %** | **100 %** (20 utterances) |
| G2P parity vs espeak-ng (voice's training phonemizer) | — | **98.4 % segmental**, 95.8 % exact | 98.5 % segmental |
| Raw round-trip CER, 12-utterance corpus | 0.62 | 0.32 (EN 0.06, Hinglish 0.58) | 0.32 (EN 0.07, Hinglish 0.57) |
| **Display CER** (post-corrected text the app shows and the LLM receives) | — | — | **0.30** (EN 0.07, Hinglish **0.54**) |
| 2-way LID accuracy (`<\|en\|>` vs `<\|hi\|>`), corpus | 87.5 % | 100 % | 100 % (held-out: 87.5 %) |
| Decoder ms / token (cache-free → KV + fused logits) | 45.4 | 15.1 | 19.9 |
| TTS RTF (INT8 ConvInteger → fp16-stored) | 0.30 | 0.11 | 0.16 |

Iteration 3 rows use 6 seeded TTS samples per utterance (`--seed 1234`); Iteration 2 used 3. The latency rows
vary with host load: a remote-desktop session shared the 4 vCPUs during the Iteration-3 run, and the models,
runtime and session options are unchanged since Iteration 2. Held-out LID: the short English sentence
*"Wake me up at 6:30 AM tomorrow."* in the Indian TTS voice was labelled Hindi in 5 of 6 samples. The
language chip can force English.

### Hinglish STT post-corrector

`HinglishCodec.displayTranscript()` turns Whisper-tiny's by-ear Roman Hinglish into the app's
canonical spelling, e.g. `Mohjee kaal subah, 7 bajyakal alarm sate karna hai` →
**`Mujhe kal subah 7 baje ka alarm set karna hai`**. It runs five deterministic table passes:
1. script normalisation;
2. split-word merges and domain spellings (`skedule`, `miting`, `forcast`, `banglor`/`bengaluru`);
3. a **Hinglish phonetic key** that folds aspiration, vowel length/quality, a post-vocalic *h*,
   `-ay/-ey` and gemination, so `mohjee`, `muje`, `mujhey` all reach `mujhe`. An unseen spelling is
   rewritten only if exactly one canonical word shares its key and it is not a real English (CMU)
   word;
4. a **dynamic-programming compound splitter** (`kalsuba` → `kal subah`, `kyahai` → `kya hai`,
   `merinext` → `meri next`);
5. **context rules**: `kaal` → `kal` beside a time word, `sate`/`seat`/`sat` → `set` after
   alarm/reminder or before *karna*, `saat baje` → `7 baje`, `baje kal alarm` → `baje ka alarm`,
   `subah, 7` → `subah 7`.

English sentences, including the English Whisper sometimes produces for Hinglish audio, pass
through unchanged. Cost: **72 µs per transcript** (host JVM; unit test asserts < 500 µs).

| Raw → display CER (paired, same transcripts) | Hinglish | English |
|---|---|---|
| Corpus (12 utterances × 6) | 0.57 → **0.54** (Δ -0.032, 95 % CI [-0.055, -0.012]; 10 better / 0 worse of 36) | 0.07 → **0.07** (Δ +0.000, 95 % CI [+0.000, +0.000]; 0 better / 0 worse of 36) |
| **Held-out** (8 new sentences × 6, written before the corrector saw them) | 0.51 → **0.49** (Δ -0.016, 95 % CI [-0.031, -0.005]; 9 better / 1 worse of 30) | 0.06 → **0.06** (Δ +0.000, 95 % CI [+0.000, +0.000]; 0 better / 0 worse of 18) |

Raw = Whisper's text as decoded; display = the same text after `displayTranscript`; both are scored in Roman script against the reference.
The corpus numbers are in-sample: the lexicon was tuned while looking at these outputs. The held-out row is the honest generalisation estimate.

Selected held-out samples (first TTS sample each; reference → raw → display):
- `H1-hi`: *Kal shaam 6 baje meeting ka reminder laga do.* → `kar sham chae bha ji, mi ting ka remandar laga do` → `kar shaam chae bha ji, meeting ka remandar laga do`
- `H2-hi`: *Parson subah 8 baje ka alarm set kar do.* → `Perseusaba 8 bajaya alarm is 8 kar 2` → `Perseusaba 8 bajaya alarm is 8 kar 2`
- `H3-hi`: *Kya kal Bangalore mein baarish hogi?* → `Keakal Bangalore me barishogi` → `Keakal Bangalore me barishogi`
- `H6-en`: *Wake me up at 6:30 AM tomorrow.* → `Wake me up 830 AM tomorrow.` → `Wake me up 830 AM tomorrow.`
- `H7-en`: *What time is my next meeting today?* → `But, time is my next meeting today.` → `But, time is my next meeting today.`
- `H8-en`: *Remind me to call the lab at 3 PM.* → `Remind me to call the lab at 3pm.` → `Remind me to call the lab at 3pm.`

### Encoder precision A/B (paired, identical seeded audio, 72 pairs)

| Encoder | Size | ORT 1.18 ms (4 thr) | CER Hinglish | CER English | Δ Hinglish vs INT8 (95 % CI) |
|---|---|---|---|---|---|
| **INT8 MatMul-only (shipped)** | 11.7 MB | 230 | 0.567 | 0.065 | — |
| INT8 per-channel | 11.8 MB | 223 | 0.571 | 0.074 | +0.004 [−0.031, +0.041] |
| fp16-stored | 16.5 MB | 243 | 0.554 | 0.059 | −0.014 [−0.056, +0.026] |
| fp32 | 32.9 MB | 243 | 0.554 | 0.059 | identical to fp16-stored |

All four load under onnxruntime 1.18.0. Higher precision gives no significant accuracy gain: the CI
spans zero, with 13 samples better and 18 worse. So **INT8 stays**: the Hindi errors come from
Whisper-tiny itself, not from quantisation. No encoder reaches 120 ms on this 2-core host, because
Whisper always encodes a 30 s window.

### Engineering findings
**Iteration 3**
1. **Groq ⇄ Gemini cascade** (above) with cool-downs, first-token timeout and provider skip; 17
   MockWebServer tests cover every failover path.
2. **Encoder precision** — fp16/fp32 vs INT8 paired A/B: no significant CER gain → INT8 kept (above).
3. **Decoding Hinglish with `<|en|>`** instead of `<|hi|>` (hoping for transliteration) is worse:
   Roman CER 0.72 vs 0.59 on the same audio (`speech_lab/decode_language_ab.py`). Rejected.
4. **STT post-corrector** (above). Most of the remaining Hinglish error is Whisper-tiny
   **translating** or hallucinating (e.g. `Today's schedule is our next meeting.`), which no
   lexicon can repair; the corrector deliberately leaves English alone.
5. **Release build**: signed with a local key; debug and release install side by side. R8 is off
   because the minified build closed at launch on the phone (`app/proguard-rules.pro` is kept for a
   later attempt).
6. **On-device fixes** (OnePlus 11R testing):
   - **TTS crash** (`ScatterND … invalid indice = 4656722024375255042`, `Expand_4969`,
     `GatherElements_4204`). The cause is onnxruntime **1.18**'s memory-pattern planner reusing stale
     buffers once VITS input lengths vary. The "index" is two float32 5.000001, the spline's tail
     bound, read as int64. It reproduces on the host in 29 of 100 varying-length runs and in 0 of 100
     with the pattern disabled, which is what the app's TTS session now does; ORT 1.31 is unaffected.
     `check_ort_compat.py --stress` guards it. Runs are also serialised, empty sentences skipped, and
     a failed sentence is retried once with deterministic durations, then logged and skipped, never
     shown in the feed. The default duration noise is now 0.2 (it was 0.8 when the tables above were
     measured).
   - **"eight"** was spelled like "at" (एट), so 28.3 was heard as "20.3" in 0 of 8 samples. With एईट
     it is heard correctly in 10 of 12.
   - **AM/PM**: a spaced `P. M.` no longer splits into two spoken sentences.
   - **Whisper non-speech output** (`1,2,3,4,5,5,5,5`, `Arrrrr`) is rejected. In a Speak-chip turn
     the chip's prompt is sent instead, labelled in the feed and excluded from WER. Impossible clock
     times next to a time word (`6:77 baje`) become the spoken hour.
   - The top-bar title stays on one line.

**Iteration 2**
1. **Runtime blocker fixed** — the Iteration-1 INT8 encoder/VITS used `ConvInteger` with int8 weights,
   which onnxruntime 1.18–1.22 cannot load. The encoder is now INT8 MatMul-only (convs fp32); the
   voice is fp16-stored. `speech_lab/check_ort_compat.py` gates every graph under onnxruntime==1.18.0.
2. **Tied-embedding logits** — a graph rewrite to `DynamicQuantizeMatMul` on one shared table:
   21.8 → 5.7 ms per cached decoder step (ORT 1.18), same file size.
3. **KV cache is exact** — 12/12 token-identical to the cache-free decoder with fp32 graphs.
4. **G2P** matches espeak-ng (the voice's training phonemizer) at 98.4 % segmental parity.
5. `gemini-2.5-flash` is retired for new keys (HTTP 404) → the Gemini rungs use the 3.5 generation.

---

## Build, install and evaluate (for IISc evaluators)

Prerequisites: JDK 17, Android SDK 34, NDK **26.1.10909125**, CMake **3.22.1** (Android Studio
Koala or newer works; the repo also builds headless with the Gradle wrapper).

```bash
git clone https://github.com/VirtualHorror/IISc-Speech_Intern-Assignment.git
cd IISc-Speech_Intern-Assignment
cat > local.properties <<'EOF'           # git-ignored; keys are compiled into BuildConfig
sdk.dir=/absolute/path/to/Android/sdk
GROQ_API_KEY=gsk_...                     # optional: cloud rungs 1-2 (Groq)
GEMINI_API_KEY=...                       # optional: cloud rungs 3-4 (Gemini)
# optional release signing (otherwise app-release.apk is signed with the debug key):
# VAANI_RELEASE_STORE_FILE=/absolute/path/release.jks
# VAANI_RELEASE_STORE_PASSWORD=...
# VAANI_RELEASE_KEY_ALIAS=...
# VAANI_RELEASE_KEY_PASSWORD=...
EOF
./gradlew testDebugUnitTest assembleDebug assembleRelease

adb install -r app/build/outputs/apk/release/app-release.apk   # "VaaniEdge"          (in.iisc.vaani)
adb install -r app/build/outputs/apk/debug/app-debug.apk       # "VaaniEdge (debug)"  (in.iisc.vaani.debug)
adb logcat -s VaaniTelemetry VaaniAgent                        # per-turn JSON telemetry
```

The two APKs have different application ids, so they install side by side. First launch extracts
~106 MB of models; wait for the status line **`ARM64 NEON + ONNX INT8 Ready`**, grant the
microphone, then use a prompt chip's **Speak** (say it) or **Loopback** (the phone speaks it to itself).
With no keys at all the app still works: replies come from the on-device `LocalResponder`.

**Switching Groq ⇄ Gemini in the app:** tap the model chip in the top bar to open **Settings**, then:
- **Primary model:** pick from Groq Qwen3.8-27B (default), Groq GPT-OSS-20B (fast/economical),
  Groq GPT-OSS-120B, Gemini 3.5 Flash-Lite or Gemini 3.5 Flash. The dialog shows the resulting
  automatic failover order.
- **Keys and switches:** keys can be pasted there; they're masked, stored app-private, and override
  the built-in ones. *Model id override* covers a provider retiring a model, and *Cloud replies* off
  forces on-device answers.
- **Save.**

Keys never enter git: `local.properties` is ignored, `BuildConfig.DEFAULT_*_API_KEY` lives only in
`app/build/`, and Settings overrides live in app-private SharedPreferences (backups disabled).
Anyone holding an APK can extract a BuildConfig key, so share APKs privately and rotate free-tier
keys after the evaluation.

## Speech lab (`speech_lab/`)

```bash
python speech_lab/prepare_models.py [--only stt,vocab,vad,tts,mel,lexicon]   # assets (all < 100 MB each)
python speech_lab/verify_closed_loop.py --runs 6                # Kotlin G2P → VITS → C++ DSP → Whisper → Kotlin post-corrector
python speech_lab/verify_closed_loop.py --ab-encoders int8=A.onnx,fp32=B.onnx --runs 6   # paired precision study
python speech_lab/decode_language_ab.py                         # <|hi|> vs <|en|> decoding experiment
python speech_lab/llm_smoke_once.py                             # EXACTLY one request per model (quota-safe)
/tmp/vaani_venv_ort118/bin/python speech_lab/check_ort_compat.py --write    # onnxruntime==1.18.0 gate
/tmp/vaani_venv_ort118/bin/python speech_lab/check_ort_compat.py --models X.onnx --runs 10 --threads 4
python speech_lab/benchmark_llm.py --runs 3                     # 12+ live calls per model: burns free-tier quota
```
`verify_closed_loop.py` builds `vaani_dsp_neon.cpp` for the host with `ziglang` and calls it through
ctypes, gets phoneme ids from `ClosedLoopFixtureTest`, and scores the display transcript produced by
`SttDisplayExportTest` (the app's own Kotlin), so the lab measures the code that ships. Optional
host-only tools: `espeakng-loader` (G2P reference), `tokenizers` (non-speech token ids).

## Known limitations
- Host numbers are indicative; the HUD/logcat give the real ones.
- **Whisper-tiny's Hinglish ceiling**: on synthetic Hinglish speech it often translates to English
  or hallucinates; the post-corrector fixes spelling, not meaning. The LLM prompt includes the raw
  ASR line and says transcripts may be noisy. A larger multilingual or Indic ASR model is the next
  accuracy step.
- LLM TTFTs are single samples from the build host (quota-safe); phone-network TTFT will differ.
- Free tiers: Groq 1,000 requests/day and 8k tokens/min per model, Gemini 3.5 Flash 20/day. The
  cascade absorbs a rate-limited rung, but heavy use can exhaust them all, leaving on-device replies.
- Barge-in is by tap (the mic is closed while speaking); streaming STT and AEC voice barge-in are
  future work.
