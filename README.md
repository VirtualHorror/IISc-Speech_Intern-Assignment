# VaaniEdge — On-Device Speech AI Voice Agent

**IISc Computational Intelligence Lab** · Android · Snapdragon 8+ Gen 1 (OnePlus 11R 5G, `arm64-v8a`)

A Hinglish + English voice agent whose speech stack runs entirely on the phone —
Silero VAD → KV-cached Whisper STT with step-0 language ID → streaming cloud LLM →
clause-level Indic VITS TTS — built from first principles on the official
**ONNX Runtime** (Microsoft, MIT). It answers the four IISc evaluation scenarios
(Bengaluru weather tomorrow, a 5 PM meeting reminder, a 7 AM alarm in Hinglish, today's
schedule) with grounded local tools, and shows a live latency HUD for every turn.
Every engine, DSP routine, codec and UI component under `in.iisc.vaani.*` is original
work; models, runtime and fonts are attributed dependencies (see [`NOTICE`](NOTICE)).

> **Status: Session 2 of 3.** `./gradlew testDebugUnitTest assembleDebug` → **BUILD
> SUCCESSFUL, 64 unit tests green**. Every shipped graph loads and runs under
> onnxruntime **1.18.0** (the APK's runtime) on the host. Speech quality and latency
> below are **measured on the build host**; the phone's own numbers appear in the in-app
> HUD and in logcat (`VaaniTelemetry`). On-device validation is the next step.

---

## Architecture

```
 mic ─▶ AudioRecord(VOICE_RECOGNITION,16k) ─▶ NEON RMS (waveform) ─▶ Silero VAD (64-sample context)
          │ 2 speech chunks ──────────────▶ onSpeechStart() ─▶ async HEAD: TCP+TLS1.3+HTTP/2 pre-warm
          │ 500 ms trailing silence ──────▶ t_end  (E2E clock starts)
          ▼
   NEON log-Mel [1,80,3000] ─▶ Whisper encoder (INT8) ─▶ decoder([SOT]) ─▶ LID: <|en|> vs <|hi|>
          (C++, zero-frame skip)                     │  cross-attn K/V computed once, reused by reference
                                                     ▼
                         decoder_with_past (KV cache, fused INT8 logits) ─▶ masked greedy ─▶ text
          ▼
   HinglishCodec.normalize/clean ─▶ LLM SSE stream (Gemini 3.5 Flash-Lite | Groq Llama) + grounded tools
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
| `codec` | `IndicPhonemizer`: Devanagari → espeak-ng-compatible IPA (aspiration `ʰ`, palatal `c ɟ`, schwa deletion, Hindi stress, nasals, gemination), Piper ids. `HinglishCodec`: language-aware numbers/times (`साढ़े चार` vs `four thirty`), curated Indian-English + Hinglish lexicons, CMUdict → Devanagari, acronyms, ASR stutter cleanup. |
| `llm` | `LlmStreamingClient` (OkHttp HTTP/2, pre-warm, Groq + Gemini SSE, retry on a fallback model for 404/429/503, per-call connection-reuse evidence), `ResponseStreamProcessor` (clause cutting, tag parsing), `PromptBuilder`, `LlmSettings`. |
| `tools` | `LocalToolManager`: Open-Meteo forecast cached at launch (0 ms lookup, offline fallback), reminders → Calendar intent, alarms → `AlarmClock.ACTION_SET_ALARM`, IISc lab schedule anchored to "now"; `LocalResponder` answers all four scenarios offline. |
| `agent`, `audio`, `runtime`, `ui` | Turn orchestration + telemetry, capture/playback pipeline, asset→engine glue with warm-up, Jetpack Compose Material 3 UI. |

### UI
Telemetry HUD (`STT (Local INT8)`, `LLM TTFT`, `TTS First-Audio`, `Total E2E Latency` with a
proportional STT | LLM | TTS latency bar), status line, language (`Auto EN/HI` / `English` /
`Hinglish`) and LLM chips, a strip with the four IISc prompts (**Speak** live or **Loopback**:
the phone synthesises the prompt, feeds that audio to STT → LLM → TTS and logs the turn), the
conversation with `[EN]`/`[HI-EN]` badges, STT ms, WER vs. the prompt, replay and a
Devanagari/IPA phonetics view, action cards (weather, alarm, reminder, schedule), a 60 fps
waveform (saffron = your voice, phosphor = the agent) and the mic button
(`Tap to Speak (Auto-VAD)` → `Listening...` → `Transcribing...` → `Streaming...` →
`Speaking (Tap to Barge-In)`). Settings: provider/model, model-id override, Groq/Gemini keys.

---

## Measured results (host: Xeon 8259CL, 4 threads; `speech_lab/evaluation_metrics.json`)

| Closed loop: Kotlin G2P → VITS → C++ Mel → Whisper | Session 1 | Session 2 |
|---|---|---|
| Phoneme coverage of `phoneme_id_map` | 99.76 % (breathy `ʱ` dropped) | **100 %** |
| G2P parity vs espeak-ng (voice's training phonemizer), 449 words | — | **98.4 % segmental**, 95.8 % exact, PER 1.3 % |
| Round-trip CER (12 utterances × 3 TTS samples) | 0.62 | **0.32** (EN 0.06, Hinglish 0.58) |
| 2-way LID accuracy (`<|en|>` vs `<|hi|>`) | 87.5 % | **100 %** |
| Decoder ms / token (cache-free → KV + fused logits) | 45.4 | **15.1** |
| C++ log-Mel vs numpy whisper reference (max abs) | — | 1.5e-5 (9.9 ms, 83 % of FFTs skipped) |
| TTS RTF (INT8 ConvInteger → fp16-stored) | 0.30 | **0.11** |

LLM (live, app prompt + the closed loop's noisy transcripts): **Gemini 3.5 Flash-Lite TTFT median
795 ms**, 5/6 tool tags; replies e.g. `[ACTION:SET_ALARM|07:00|Morning Alarm]` + *"Kal subah 7:00 AM
ka alarm set ho gaya hai."* Groq was not benchmarked (no key at build time).

### Engineering findings this session
1. **Runtime blocker fixed** — the Session-1 INT8 encoder and INT8 VITS use `ConvInteger` with int8
   weights, which onnxruntime 1.18–1.22 cannot load (`NOT_IMPLEMENTED`): STT and TTS would have
   failed on the phone. The encoder is now INT8 MatMul-only (convs fp32); the voice is fp16-stored.
   `speech_lab/check_ort_compat.py` gates every graph under onnxruntime==1.18.0.
2. **Tied-embedding logits** — the community INT8 decoders de-quantised the 51865×384 embedding table
   on *every* step (16.7 of 28.7 ms). A graph rewrite to `DynamicQuantizeMatMul` on one shared table:
   29.1 → 9.5 ms per cached step (ORT 1.31), 21.8 → 5.7 ms (ORT 1.18), same file size.
3. **KV cache is exact** — 12/12 token-identical to the cache-free decoder with fp32 graphs (INT8
   graphs differ only through dynamic activation scales).
4. **G2P** — breathy consonants now use `ʰ` (as espeak-ng did for the voice's training data), plus
   `c/ɟ`, plain dentals, medial schwa deletion, stress, nasals, Piper's per-code-point padding.
5. **Precision study** — fp16-stored encoder+decoders lower Hinglish CER 0.61 → 0.54 (English
   unchanged); INT8 is kept for latency. Suppressing Latin tokens or Devanagari prompting made
   Whisper-tiny worse (CER 0.98 / 1.08) and was rejected.
6. **`gemini-2.5-flash` is retired for new keys** (HTTP 404) — the app uses Gemini 3.5 Flash-Lite
   (`thinkingLevel=minimal`; thinking otherwise eats the 160-token budget), with 3.5 Flash selectable
   and automatic fallback on 404/429/503.

---

## Build & run (Ubuntu headless)

```bash
source $HOME/.vaani_env                       # JDK 17, Android SDK/NDK (see HANDOFF notes)
cat >> local.properties <<'EOF'               # git-ignored; compiled into BuildConfig
GEMINI_API_KEY=...
GROQ_API_KEY=                                  # optional; can also be pasted in-app
EOF
./gradlew testDebugUnitTest assembleDebug     # -> app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb logcat -s VaaniTelemetry VaaniAgent       # per-turn JSON telemetry from the phone
```

Keys never enter git: `local.properties` is ignored, `BuildConfig.DEFAULT_*_API_KEY` lives only in
`app/build/`, and runtime overrides live in app-private SharedPreferences (backups disabled).
Anyone holding the APK can extract a BuildConfig key — keep the APK private.

## Speech lab (`speech_lab/`)

```bash
python speech_lab/prepare_models.py [--only stt,vocab,vad,tts,mel,lexicon]  # assets (all < 100 MB each)
python speech_lab/verify_closed_loop.py --runs 3   # Kotlin fixture (via Gradle) → VITS → C++ DSP → Whisper
python speech_lab/benchmark_llm.py --runs 3        # live TTFT + tool-tag compliance (uses local.properties)
python speech_lab/benchmark_models.py              # tensor signatures + sizes
/tmp/vaani_venv_ort118/bin/python speech_lab/check_ort_compat.py --write   # onnxruntime==1.18.0 gate
```
`verify_closed_loop.py` builds `vaani_dsp_neon.cpp` for the host with `ziglang` and calls it through
ctypes, and gets phoneme ids from `ClosedLoopFixtureTest` — the lab measures the code that ships.
Optional host-only tools: `espeakng-loader` (G2P reference), `tokenizers` (non-speech token ids).

## Known limitations / Session 3
- **Not yet run on the phone.** Host numbers are indicative; the HUD/logcat give the real ones.
- **Whisper-tiny's Hinglish ceiling** (CER ≈ 0.58 on synthetic speech; often romanised or
  paraphrased). The LLM is told transcripts may be noisy. Next knobs: the fp16-stored encoder
  (+5 MB), then whisper-base.
- Groq model ids are unverified without a key (use the Settings model-id override if retired).
- Session 3: on-device profiling (thread placement, XNNPACK for fp32 ops), barge-in by voice with
  AEC, KV-cached streaming STT, release build with R8.
