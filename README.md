# VaaniEdge — On-Device Speech AI

**IISc Computational Intelligence Lab** · Android · Snapdragon 8+ Gen 1 (OnePlus 11R 5G, `arm64-v8a`)

A fully on-device, offline Hindi/Hinglish + English speech stack: streaming VAD →
Whisper STT → Hinglish normalization, and Hinglish/English → Indic neural TTS — built
from first principles on top of the official **ONNX Runtime** (Microsoft, MIT). Every
engine, DSP routine, and codec under `in.iisc.vaani.*` is original work; the ML models
and the runtime are used as attributed dependencies (see [`NOTICE`](NOTICE)).

> **Status: Session 1 of 3 — foundation complete.** Headless host build is green
> (`./gradlew testDebugUnitTest assembleDebug` → `BUILD SUCCESSFUL`, 17/17 unit tests
> passing). Engines are compiled and wired against the real model tensor signatures;
> on-device runtime validation and the conversational UI/LLM layer are Sessions 2–3.

---

## Architecture

```
            ┌──────────────────────── capture ────────────────────────┐
  mic ─▶ AudioRecord(VOICE_RECOGNITION,16k) ─▶ NEON pre-emphasis+RMS ─▶ Silero VAD
            │                                                            │ 320ms endpoint
            │                                                            ▼
            │   NEON log-Mel [1,80,3000] ─▶ Whisper encoder ─▶ greedy decode
            │        (libvaani_dsp_neon.so)      (ONNX Runtime)   + logit script-mask
            │                                                            │
            │                                                   HinglishCodec.normalize
            │                                                            ▼
            │                                               "Mujhe kal subah 7 baje…"
            └──────────────────────── playback ───────────────────────┘
   clause ─▶ HinglishCodec.toAcousticDevanagari ─▶ IndicPhonemizer(G2P) ─▶ VITS (ONNX)
          ─▶ NEON float→PCM16 ─▶ AudioTrack(MODE_STREAM, LOW_LATENCY)  [barge-in]
```

### Components (all original, package `in.iisc.vaani.*`)

| Module | What it does |
|---|---|
| `dsp.NeonAudioProcessor` + `cpp/vaani_dsp_neon.cpp` | Custom C++17 ARM64 NEON DSP: pre-emphasis, RMS envelope, noise gate, **Whisper-exact log-Mel** (radix-2+DFT FFT, NEON Mel matmul), saturating float→PCM16. |
| `neural.VaaniVadEngine` | Silero VAD over ONNX Runtime, stateful, 320 ms trailing-silence auto-endpointing. |
| `neural.VaaniSttEngine` + `VaaniVocab` | Whisper encode→greedy-decode with **logit-level script masking** (no drift to Urdu/Cyrillic/CJK) + byte-level BPE detokenization. |
| `neural.VaaniTtsEngine` + `VaaniTtsConfig` | VITS synthesis over ONNX Runtime (`scales=[0.667,1.0,0.8]`). |
| `codec.HinglishCodec` | Devanagari ↔ Roman Hinglish/English (+ numeric/time/temperature → Devanagari) in <0.5 ms. |
| `codec.IndicPhonemizer` | Pure-Kotlin Devanagari+English G2P → IPA with **Hindi schwa deletion** → Piper phoneme-id layout. |
| `audio.AudioPipelineManager` | Capture → endpoint → STT → normalize, and micro-clause streaming TTS with instant barge-in. |

---

## Build (Ubuntu headless)

Requires JDK 17, Android SDK platform-34, build-tools 34.0.0, NDK 26.1.10909125, CMake 3.22.1.
`local.properties` points at the SDK (git-ignored).

```bash
./gradlew testDebugUnitTest assembleDebug
# → app/build/outputs/apk/debug/app-debug.apk   (arm64-v8a only)
```

The APK ships `libvaani_dsp_neon.so` (our DSP) + ONNX Runtime, and the `.onnx`/`.bin`
model assets are stored **uncompressed** for mmap.

## Model pipeline (`speech_lab/`)

```bash
python speech_lab/prepare_models.py     # download, INT8-quantize, Mel export, vocab calibration
python speech_lab/benchmark_models.py   # tensor signatures + compression + projected RTF
```

- Produces `app/src/main/assets/models/` (Whisper tiny INT8 enc/dec, Silero VAD, VITS INT8,
  Mel filterbank `.bin`, script-calibration `vaani_vocab.json`, TTS config). Total ≈ 61 MB,
  every file well under GitHub's 100 MB limit.
- INT8 reductions: STT encoder 69%, decoder 74%, VITS 71% (`speech_lab/evaluation_metrics.json`).
- RTF figures there are **analytical projections** for the Snapdragon 8+ Gen 1 — to be
  validated on the OnePlus 11R in a later session.

## IISc evaluation prompts (unit-tested, both directions)

```
weather forecast for Bangalore
reminder for my meeting at 5 PM
Mujhe kal subah 7 baje ka alarm set karna hai
Aaj ka schedule kya hai, aur meri next meeting kab hai
```

## Attribution

Runtime: ONNX Runtime (Microsoft, MIT). Models: OpenAI Whisper (MIT), Silero VAD (MIT),
Piper/VITS `hi_IN` voice (rhasspy/piper-voices). Full details in [`NOTICE`](NOTICE).

## Roadmap

- **Session 1 (done):** headless build, NEON DSP, ONNX engines, Hinglish codec + G2P, model pipeline, tests.
- **Session 2:** on-device validation, KV-cache decoder, XNNPACK/NNAPI EP, LLM HTTP/2 pre-warm on `onSpeechStart()`.
- **Session 3:** conversational UI, end-to-end latency tuning, barge-in polish.
