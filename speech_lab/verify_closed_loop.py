#!/usr/bin/env python3
# =============================================================================
#  VaaniEdge — Closed-Loop Acoustic Verification  (TTS -> STT round trip)
#  IISc Computational Intelligence Lab
#
#  Measures the speech stack exactly as it ships:
#
#    corpus text --[Kotlin HinglishCodec + IndicPhonemizer]--> phoneme ids
#      (exported by app/src/test/.../ClosedLoopFixtureTest.kt -> no Python G2P port)
#    --> vaani_tts_vits.fp16.onnx (22.05 kHz)
#    --> resample to 16 kHz (our C++ Kaiser-sinc, checked against scipy)
#    --> log-Mel [1,80,3000] from app/src/main/cpp/vaani_dsp_neon.cpp (host build,
#        ctypes), checked against a numpy re-implementation of whisper.audio
#    --> vaani_stt_encoder.int8.onnx
#    --> step-0 2-way language ID (<|en|> vs <|hi|>) on the decoder's first logits
#    --> greedy decode, cache-free (Session 1) vs KV-cached decoder_with_past (Session 2)
#    --> transcripts, CER/WER, RTF, ms/token  ->  speech_lab/evaluation_metrics.json
#    --> the app's display pipeline (HinglishCodec.displayTranscript, run through Gradle:
#        app/src/test/.../SttDisplayExportTest.kt) -> "display" CER/WER of the exact text
#        the feed shows and the LLM receives, against the reference Roman text
#
#  --ab-encoders int8=a.onnx,fp32=b.onnx  runs a paired precision study instead: every
#  encoder transcribes the SAME seeded TTS audio, so CER differences are not VITS noise
#  (-> "encoder_precision_ab").
#
#  Optional: if `espeakng_loader` is installed, the Kotlin G2P is scored against
#  espeak-ng (the phonemizer the Piper hi_IN voice was trained with). espeak-ng is
#  used ONLY as a host-side measuring stick; nothing from it ships in the APK.
#
#  All timings are measured on the HOST CPU (see "host" in the output); the app's
#  HUD reports the live on-device numbers.
#
#  Usage:  python speech_lab/verify_closed_loop.py [--runs 3] [--no-gradle] [--seed 1234]
#          python speech_lab/verify_closed_loop.py --ab-encoders int8=A.onnx,fp16w=B.onnx --runs 6
#          python speech_lab/verify_closed_loop.py --rescore-display   # post-corrector change only
# =============================================================================
import argparse
import ctypes
import datetime as dt
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
import wave

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
ASSETS = os.path.join(REPO, "app", "src", "main", "assets", "models")
BUILD = os.path.join(REPO, "app", "build", "vaani")
FIXTURE = os.path.join(BUILD, "closed_loop_fixture.json")
RAW_TX = os.path.join(BUILD, "closed_loop_raw_transcripts.json")
DISPLAY_TX = os.path.join(BUILD, "closed_loop_display_transcripts.json")
CPP_SRC = os.path.join(REPO, "app", "src", "main", "cpp", "vaani_dsp_neon.cpp")
HOST_DSP = os.path.join(BUILD, "libvaani_dsp_host.so")
METRICS = os.path.join(HERE, "evaluation_metrics.json")
WITH_PAST_CANDIDATES = [
    os.path.join(ASSETS, "vaani_stt_decoder_with_past.int8.onnx"),
    "/tmp/raw_models/onnx/decoder_with_past_model_int8.onnx",
]

TTS_SR, STT_SR = 22050, 16000
N_FFT, HOP, N_MELS, N_FRAMES = 400, 160, 80, 3000
N_SAMPLES = HOP * N_FRAMES
SENTENCE_SILENCE_S = 0.2           # Piper's default inter-sentence silence
TTS_SCALES = [0.667, 1.0, 0.8]     # vaani_tts_config.json "inference"


def log(*a):
    print("[closed-loop]", *a, flush=True)


# ---------------------------------------------------------------------------
#  Fixture (exact Kotlin front-end) + host build of the C++ DSP
# ---------------------------------------------------------------------------
def regenerate_fixture():
    cmd = ("source $HOME/.vaani_env 2>/dev/null; ./gradlew :app:testDebugUnitTest "
           "--tests 'in.iisc.vaani.lab.ClosedLoopFixtureTest' -q --console=plain")
    log("exporting Kotlin G2P fixture via Gradle ...")
    r = subprocess.run(["bash", "-c", cmd], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        log("  (ClosedLoopFixtureTest reported failures — fixture still written; see report)")


def kotlin_display(items):
    """[{key, raw, lang}] -> {key: {display, us}} computed by the app's own Kotlin
    HinglishCodec.displayTranscript (SttDisplayExportTest), never by a Python port."""
    os.makedirs(BUILD, exist_ok=True)
    with open(RAW_TX, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    if os.path.exists(DISPLAY_TX):
        os.remove(DISPLAY_TX)
    cmd = ("source $HOME/.vaani_env 2>/dev/null; ./gradlew :app:testDebugUnitTest "
           "--tests 'in.iisc.vaani.lab.SttDisplayExportTest' --rerun -q --console=plain")
    log(f"normalising {len(items)} transcripts with the Kotlin display pipeline via Gradle ...")
    r = subprocess.run(["bash", "-c", cmd], cwd=REPO, capture_output=True, text=True)
    os.remove(RAW_TX)
    if r.returncode != 0 or not os.path.exists(DISPLAY_TX):
        log("  Kotlin display export failed:", (r.stdout + r.stderr)[-800:])
        return None
    return {o["key"]: o for o in json.load(open(DISPLAY_TX, encoding="utf-8"))}


def build_host_dsp():
    if os.path.exists(HOST_DSP) and os.path.getmtime(HOST_DSP) >= os.path.getmtime(CPP_SRC):
        return HOST_DSP
    java_home = os.environ.get("JAVA_HOME", os.path.expanduser("~/jdk17"))
    cmd = [sys.executable, "-m", "ziglang", "c++", "-shared", "-fPIC", "-O3", "-ffast-math",
           "-std=c++17", f"-I{java_home}/include", f"-I{java_home}/include/linux",
           CPP_SRC, "-o", HOST_DSP]
    log("building host DSP library from vaani_dsp_neon.cpp (zig c++) ...")
    os.makedirs(BUILD, exist_ok=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log("  host DSP build failed:", r.stderr[-400:])
        return None
    return HOST_DSP


class HostDsp:
    """ctypes view of the C ABI exported by vaani_dsp_neon.cpp."""

    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        fp = ctypes.POINTER(ctypes.c_float)
        self.lib.vaani_log_mel.argtypes = [fp, ctypes.c_int, fp] + [ctypes.c_int] * 5 + [fp]
        self.lib.vaani_log_mel.restype = ctypes.c_int
        self.lib.vaani_resample_len.argtypes = [ctypes.c_int] * 3
        self.lib.vaani_resample_len.restype = ctypes.c_int
        self.lib.vaani_resample.argtypes = [fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, fp, ctypes.c_int]
        self.lib.vaani_resample.restype = ctypes.c_int

    @staticmethod
    def _p(a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def log_mel(self, pcm, filters):
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        out = np.empty(N_MELS * N_FRAMES, dtype=np.float32)
        fft_frames = self.lib.vaani_log_mel(self._p(pcm), len(pcm), self._p(filters), N_MELS,
                                            filters.shape[1], N_FFT, HOP, N_FRAMES, self._p(out))
        return out.reshape(1, N_MELS, N_FRAMES), fft_frames

    def resample(self, x, in_sr, out_sr):
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = self.lib.vaani_resample_len(len(x), in_sr, out_sr)
        y = np.empty(n, dtype=np.float32)
        wrote = self.lib.vaani_resample(self._p(x), len(x), in_sr, out_sr, self._p(y), n)
        return y[:wrote]


def whisper_log_mel_reference(audio, filters):
    """numpy transcription of whisper.audio.log_mel_spectrogram(pad_or_trim(audio))."""
    x = np.zeros(N_SAMPLES, dtype=np.float64)
    n = min(len(audio), N_SAMPLES)
    x[:n] = audio[:n]
    xp = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)      # periodic Hann
    n_frames = 1 + (len(xp) - N_FFT) // HOP                                  # 3001
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(xp[idx] * window, axis=1)
    power = (np.abs(spec) ** 2).T[:, :-1]                                     # drop last frame
    mel = filters.astype(np.float64) @ power
    log_spec = np.log10(np.maximum(mel, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)[None]


# ---------------------------------------------------------------------------
#  ONNX Runtime sessions (same options the Kotlin engines set)
# ---------------------------------------------------------------------------
def make_session(path, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = threads
    so.add_session_config_entry("session.intra_op.allow_spinning", "1")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
#  Whisper vocabulary / detokenisation / logit masks
# ---------------------------------------------------------------------------
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = bytes_to_unicode()


class Vocab:
    def __init__(self, path):
        v = json.load(open(path, encoding="utf-8"))
        self.size = v["vocab_size"]
        self.tokens = v["tokens"]
        st = v["special_tokens"]
        self.eot, self.sot = st["endoftext"], st["startoftranscript"]
        self.transcribe, self.notimestamps = st["transcribe"], st["notimestamps"]
        self.ts_begin = st["timestamp_begin"]
        self.lang = v["language_tokens"]
        self.script_suppressed = np.array(v["suppressed_token_ids"], dtype=np.int64)
        self.non_speech = np.array(v.get("non_speech_token_ids", []), dtype=np.int64)
        self.blank = v.get("blank_token_id", 220)

        # Session 1 mask: script calibration + timestamps only.
        self.mask_v1 = np.zeros(self.size, dtype=bool)
        self.mask_v1[self.script_suppressed] = True
        self.mask_v1[self.ts_begin:] = True
        # Session 2 mask: + Whisper non-speech symbols + every special token except EOT.
        self.mask_v2 = self.mask_v1.copy()
        self.mask_v2[self.non_speech] = True
        self.mask_v2[self.eot + 1:] = True
        self.mask_v2_first = self.mask_v2.copy()          # suppress_blank on the 1st step
        self.mask_v2_first[[self.eot, self.blank]] = True

    def decode(self, ids):
        s = "".join(self.tokens[i] for i in ids if 0 <= i < self.eot)
        raw = bytes(BYTE_DECODER[c] for c in s if c in BYTE_DECODER)
        return raw.decode("utf-8", errors="replace").replace("�", "").strip()


def repeating_tail(tokens, max_period=8, min_repeats=4):
    """True if the tail of `tokens` is one short pattern repeated (Whisper loop)."""
    for p in range(1, max_period + 1):
        need = p * min_repeats
        if len(tokens) >= need:
            tail = tokens[-need:]
            if all(tail[i] == tail[i % p] for i in range(need)):
                return True
    return False


class WhisperOnnx:
    def __init__(self, enc_path, dec_path, past_path, vocab, threads):
        self.enc = make_session(enc_path, threads)
        self.dec = make_session(dec_path, threads)
        self.past = make_session(past_path, threads) if past_path else None
        self.vocab = vocab
        self.dec_out = [o.name for o in self.dec.get_outputs()]
        self.past_out = [o.name for o in self.past.get_outputs()] if self.past else []

    def encode(self, mel):
        t0 = time.perf_counter()
        hidden = self.enc.run_with_ort_values(
            ["last_hidden_state"], {"input_features": ort.OrtValue.ortvalue_from_numpy(mel)})[0]
        return hidden, (time.perf_counter() - t0) * 1e3

    # ---- Session 1 algorithm: re-feed the whole prefix every step -------------
    # Same masks / guards as decode_kv, so identical output proves the KV cache exact.
    def decode_cache_free(self, hidden_np, lang, audio_sec, max_tokens=224):
        v = self.vocab
        toks = [v.sot, v.lang[lang], v.transcribe, v.notimestamps]
        prompt = len(toks)
        cap = min(max_tokens, 32 + int(audio_sec * 20))
        t0 = time.perf_counter()
        for i in range(cap):
            logits = self.dec.run(["logits"], {
                "input_ids": np.array([toks], dtype=np.int64),
                "encoder_hidden_states": hidden_np})[0][0, -1]
            logits[v.mask_v2_first if i == 0 else v.mask_v2] = -np.inf
            nxt = int(np.argmax(logits))
            if nxt == v.eot:
                break
            toks.append(nxt)
            if repeating_tail(toks[prompt:]):
                break
        ms = (time.perf_counter() - t0) * 1e3
        gen = toks[prompt:]
        return gen, ms, len(gen) + 1

    # ---- Session 2: step-0 LID + KV cache, encoder K/V computed exactly once --
    def decode_kv(self, hidden, lang_mode, audio_sec, max_tokens=224):
        v = self.vocab
        t0 = time.perf_counter()
        ov = ort.OrtValue.ortvalue_from_numpy
        outs = self.dec.run_with_ort_values(self.dec_out, {
            "input_ids": ov(np.array([[v.sot]], dtype=np.int64)),
            "encoder_hidden_states": hidden})
        step0 = outs[0].numpy()[0, -1]
        l_en, l_hi = float(step0[v.lang["en"]]), float(step0[v.lang["hi"]])
        p_hi = 1.0 / (1.0 + np.exp(-(l_hi - l_en)))
        detected = "hi" if l_hi > l_en else "en"
        lid_ms = (time.perf_counter() - t0) * 1e3
        lang = detected if lang_mode == "auto" else lang_mode

        kv = {}
        for name, val in zip(self.dec_out, outs):
            if name.startswith("present."):
                kv[name.replace("present.", "past_key_values.")] = val
        enc_kv = {k: val for k, val in kv.items() if ".encoder." in k}    # reused every step
        dec_kv = {k: val for k, val in kv.items() if ".decoder." in k}

        def step(tok):
            nonlocal dec_kv
            feeds = {"input_ids": ov(np.array([[tok]], dtype=np.int64))}
            feeds.update(dec_kv)
            feeds.update(enc_kv)
            o = self.past.run_with_ort_values(self.past_out, feeds)
            dec_kv = {n.replace("present.", "past_key_values."): val
                      for n, val in zip(self.past_out, o) if n.startswith("present.")}
            return o[0].numpy()[0, -1]

        for forced in (v.lang[lang], v.transcribe):
            step(forced)
        logits = step(v.notimestamps)
        cap = min(max_tokens, 32 + int(audio_sec * 20))
        gen = []
        steps = 4
        for i in range(cap):
            mask = v.mask_v2_first if i == 0 else v.mask_v2
            logits = logits.copy()
            logits[mask] = -np.inf
            nxt = int(np.argmax(logits))
            if nxt == v.eot:
                break
            gen.append(nxt)
            if repeating_tail(gen):
                break
            logits = step(nxt)
            steps += 1
        ms = (time.perf_counter() - t0) * 1e3
        return {"tokens": gen, "lang": lang, "detected": detected, "p_hi": float(p_hi),
                "lid_ms": lid_ms, "decode_ms": ms, "decoder_calls": steps}


# ---------------------------------------------------------------------------
#  Text normalisation + error rates
# ---------------------------------------------------------------------------
HI_NUM = {0: "शून्य", 1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच", 6: "छह", 7: "सात", 8: "आठ",
          9: "नौ", 10: "दस", 11: "ग्यारह", 12: "बारह", 13: "तेरह", 14: "चौदह", 15: "पंद्रह",
          16: "सोलह", 17: "सत्रह", 18: "अठारह", 19: "उन्नीस", 20: "बीस", 21: "इक्कीस", 22: "बाईस",
          23: "तेईस", 24: "चौबीस", 25: "पच्चीस", 26: "छब्बीस", 27: "सत्ताईस", 28: "अट्ठाईस",
          29: "उनतीस", 30: "तीस", 40: "चालीस", 45: "पैंतालीस", 50: "पचास"}
EN_NUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
EN_TENS = {"thirty": 30, "forty": 40, "fifty": 50}
DEV_PUNCT = re.compile(r"[।॥,.?!;:\"'()\-]")
LAT_PUNCT = re.compile(r"[^a-z0-9\s]")


def is_devanagari(s):
    dev = sum(1 for c in s if "ऀ" <= c <= "ॿ")
    lat = sum(1 for c in s if c.isascii() and c.isalpha())
    return dev > lat


def norm_dev(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace("़", "").replace("ँ", "ं")
    s = "".join(chr(ord(c) - 0x0966 + 0x30) if "०" <= c <= "९" else c for c in s)
    s = re.sub(r"\d+", lambda m: HI_NUM.get(int(m.group()), m.group()), s)
    s = DEV_PUNCT.sub(" ", s)
    return " ".join(s.split())


def norm_lat(s):
    s = s.lower().replace("p.m.", "pm").replace("a.m.", "am")
    s = re.sub(r"(\d)(am|pm)\b", r"\1 \2", s)
    s = LAT_PUNCT.sub(" ", s)
    words = []
    for w in s.split():
        if w in EN_NUM:
            words.append(str(EN_NUM[w]))
        elif w in EN_TENS:
            words.append(str(EN_TENS[w]))
        else:
            words.append(w)
    merged = []
    for w in words:   # "20 8" -> "28"
        if merged and merged[-1].isdigit() and int(merged[-1]) in (20, 30, 40, 50) \
                and w.isdigit() and int(w) < 10:
            merged[-1] = str(int(merged[-1]) + int(w))
        else:
            merged.append(w)
    return " ".join(merged)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def score(hyp, utt):
    """CER/WER in the hypothesis' script: Devanagari vs the acoustic script we spoke,
    Latin vs the original Roman/English text."""
    if is_devanagari(hyp):
        ref, h, space = norm_dev(utt["acoustic_devanagari"]), norm_dev(hyp), "devanagari"
    else:
        ref, h, space = norm_lat(utt["text"]), norm_lat(hyp), "latin"
    cer = edit_distance(list(ref.replace(" ", "")), list(h.replace(" ", ""))) / max(1, len(ref.replace(" ", "")))
    wer = edit_distance(ref.split(), h.split()) / max(1, len(ref.split()))
    return {"cer": round(cer, 4), "wer": round(wer, 4), "space": space, "ref_norm": ref, "hyp_norm": h}


def score_display(display, utt):
    """CER/WER of the app's display transcript against the reference Roman text."""
    ref, h = norm_lat(utt["text"]), norm_lat(display)
    cer = edit_distance(list(ref.replace(" ", "")), list(h.replace(" ", ""))) / max(1, len(ref.replace(" ", "")))
    wer = edit_distance(ref.split(), h.split()) / max(1, len(ref.split()))
    return {"cer": round(cer, 4), "wer": round(wer, 4), "ref_norm": ref, "hyp_norm": h}


def by_lang(values_with_lang):
    out = {}
    for lang in ("en", "hi"):
        v = [x for x, lg in values_with_lang if lg == lang]
        out[lang] = summarize(v)
    out["all"] = summarize([x for x, _ in values_with_lang])
    return out


def synthesize(tts, utt):
    """Piper layout: one VITS call per sentence, 0.2 s gaps, peak-normalised to 0.95."""
    t0 = time.perf_counter()
    pieces, first_audio_ms = [], None
    for ids in utt["sentence_ids"]:
        x = np.array([ids], dtype=np.int64)
        y = tts.run(None, {"input": x, "input_lengths": np.array([x.shape[1]], np.int64),
                           "scales": np.array(TTS_SCALES, np.float32)})[0].reshape(-1)
        if first_audio_ms is None:
            first_audio_ms = (time.perf_counter() - t0) * 1e3
        pieces += [y, np.zeros(int(SENTENCE_SILENCE_S * TTS_SR), np.float32)]
    tts_ms = (time.perf_counter() - t0) * 1e3
    audio22 = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, np.float32)
    peak = float(np.max(np.abs(audio22))) if audio22.size else 0.0
    return audio22 * (0.95 / max(peak, 1e-3)), tts_ms, first_audio_ms, peak


# ---------------------------------------------------------------------------
#  espeak-ng parity (optional host-side measuring stick)
# ---------------------------------------------------------------------------
class Espeak:
    def __init__(self):
        import espeakng_loader as el
        self.lib = ctypes.cdll.LoadLibrary(el.get_library_path())
        self.lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self.lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        self.lib.espeak_TextToPhonemes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
        self.lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        self.lib.espeak_Initialize(0x02, 0, el.get_data_path().encode(), 0)
        if self.lib.espeak_SetVoiceByName(b"hi") != 0:
            raise RuntimeError("espeak-ng hi voice unavailable")

    def word(self, w):
        buf = ctypes.create_string_buffer(w.encode("utf-8"))
        ptr = ctypes.c_void_p(ctypes.addressof(buf))
        out = []
        while ptr.value:
            r = self.lib.espeak_TextToPhonemes(ctypes.byref(ptr), 1, 0x02)
            if r:
                out.append(r.decode("utf-8"))
        return unicodedata.normalize("NFD", " ".join(out).strip())


def g2p_parity(fixture):
    words = fixture.get("word_probe") or []
    if not words:
        return None
    try:
        es = Espeak()
    except Exception as e:  # noqa: BLE001
        log("  espeak-ng not available:", type(e).__name__, e)
        return None
    strip = lambda s: re.sub("[ˈˌ]", "", s)                      # noqa: E731
    exact = exact_seg = 0
    dist = total = 0
    worst = []
    for item in words:
        ours = unicodedata.normalize("NFD", item["ipa"])
        ref = es.word(item["word"])
        exact += ours == ref
        exact_seg += strip(ours) == strip(ref)
        d = edit_distance(list(ours), list(ref))
        dist += d
        total += len(ref)
        if strip(ours) != strip(ref):
            worst.append((d, item["word"], ours, ref))
    worst.sort(reverse=True)
    return {
        "reference": "espeak-ng hi (piper training phonemizer), host-side only",
        "words": len(words),
        "exact_match_pct": round(100.0 * exact / len(words), 2),
        "segmental_match_pct": round(100.0 * exact_seg / len(words), 2),
        "phoneme_error_rate_pct": round(100.0 * dist / max(1, total), 2),
        "top_mismatches": [{"word": w, "kotlin": o, "espeak": r} for _, w, o, r in worst[:25]],
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def save_wav(path, audio, sr):
    pcm = np.clip(audio, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767.0).astype("<i2").tobytes())


def summarize(values):
    values = [float(x) for x in values if x is not None]
    if not values:
        return None
    sig = lambda x: float(f"{x:.5g}")                  # noqa: E731  (keeps 1e-7 errors visible)
    return {"mean": sig(statistics.fmean(values)), "median": sig(statistics.median(values)),
            "min": sig(min(values)), "max": sig(max(values))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="TTS samples per utterance (VITS is stochastic)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-gradle", action="store_true", help="reuse the existing fixture")
    ap.add_argument("--out", default=METRICS)
    ap.add_argument("--label", default="session2")
    ap.add_argument("--encoder", default=os.path.join(ASSETS, "vaani_stt_encoder.int8.onnx"))
    ap.add_argument("--tts", default=os.path.join(ASSETS, "vaani_tts_vits.fp16.onnx"))
    ap.add_argument("--seed", type=int, default=1234, help="ORT RNG seed (VITS noise) for reproducible audio")
    ap.add_argument("--ab-encoders", default="", help="name=path,... : paired encoder precision study")
    ap.add_argument("--rescore-display", action="store_true",
                    help="re-run only the Kotlin display pass on the transcripts of the last full run")
    args = ap.parse_args()
    if args.rescore_display:
        return rescore_display(args)
    ort.set_seed(args.seed)

    if not args.no_gradle or not os.path.exists(FIXTURE):
        regenerate_fixture()
    fixture = json.load(open(FIXTURE, encoding="utf-8"))
    log(f"fixture: {len(fixture['utterances'])} utterances, "
        f"phoneme coverage {fixture['phoneme_coverage_pct']:.2f}% "
        f"({fixture['phoneme_codepoints_dropped']} dropped)")

    filters = np.fromfile(os.path.join(ASSETS, "vaani_mel_filters.bin"), dtype="<f4").reshape(N_MELS, -1)
    dsp_path = build_host_dsp()
    dsp = HostDsp(dsp_path) if dsp_path else None

    vocab = Vocab(os.path.join(ASSETS, "vaani_vocab.json"))
    past_path = next((p for p in WITH_PAST_CANDIDATES if os.path.exists(p)), None)
    stt = WhisperOnnx(args.encoder, os.path.join(ASSETS, "vaani_stt_decoder.int8.onnx"), past_path,
                      vocab, args.threads)
    tts = make_session(args.tts, args.threads)

    # Warm-up (first-run allocation / kernel selection excluded from timings).
    warm = np.zeros(STT_SR, dtype=np.float32)
    hidden, _ = stt.encode(whisper_log_mel_reference(warm, filters))
    stt.decode_kv(hidden, "en", 1.0, max_tokens=2)
    tts.run(None, {"input": np.array([[1, 0, 14, 0, 2]], np.int64),
                   "input_lengths": np.array([5], np.int64), "scales": np.array(TTS_SCALES, np.float32)})

    if args.ab_encoders:
        return run_encoder_ab(args, fixture, filters, dsp, stt, tts)

    wav_dir = os.path.join(BUILD, "closed_loop_wavs")
    os.makedirs(wav_dir, exist_ok=True)
    rows = []
    mel_err, resample_snr, fft_frac = [], [], []
    for utt in fixture["utterances"]:
        runs = []
        for r in range(args.runs):
            # ---- TTS (sentence by sentence, Piper layout) ----
            audio22, tts_ms, first_audio_ms, peak = synthesize(tts, utt)
            audio_sec = len(audio22) / TTS_SR

            # ---- resample 22.05k -> 16k (C++ sinc; scipy for reference) ----
            ref16 = resample_poly(audio22, 320, 441).astype(np.float32)
            if dsp:
                ours16 = dsp.resample(audio22, TTS_SR, STT_SR)
                n = min(len(ours16), len(ref16))
                err = ours16[:n] - ref16[:n]
                resample_snr.append(float(10 * np.log10(np.sum(ref16[:n] ** 2) / max(np.sum(err ** 2), 1e-20))))
                audio16 = ours16
            else:
                audio16 = ref16
            if r == 0:
                save_wav(os.path.join(wav_dir, f"{utt['id']}.wav"), audio22, TTS_SR)

            # ---- log-Mel: C++ (shipped) vs numpy Whisper reference ----
            t1 = time.perf_counter()
            mel_ref = whisper_log_mel_reference(audio16, filters)
            ref_ms = (time.perf_counter() - t1) * 1e3
            if dsp:
                t1 = time.perf_counter()
                mel, fft_frames = dsp.log_mel(audio16, filters)
                mel_ms = (time.perf_counter() - t1) * 1e3
                mel_err.append(float(np.max(np.abs(mel - mel_ref))))
                fft_frac.append(fft_frames / N_FRAMES)
            else:
                mel, mel_ms = mel_ref, ref_ms

            # ---- encoder + LID + decode (auto and forced language) ----
            hidden, enc_ms = stt.encode(mel)
            auto = stt.decode_kv(hidden, "auto", audio_sec)
            forced = auto if auto["lang"] == utt["lang"] else stt.decode_kv(hidden, utt["lang"], audio_sec)
            cf_tokens, cf_ms, cf_calls = stt.decode_cache_free(hidden.numpy(), forced["lang"], audio_sec)
            run = {
                "tts_ms": tts_ms, "tts_first_audio_ms": first_audio_ms, "audio_sec": audio_sec,
                "tts_rtf": (tts_ms / 1e3) / audio_sec if audio_sec else None, "raw_peak": peak,
                "mel_ms": mel_ms, "mel_numpy_ms": ref_ms, "enc_ms": enc_ms,
                "auto": auto, "forced": forced,
                "auto_text": vocab.decode(auto["tokens"]), "forced_text": vocab.decode(forced["tokens"]),
                "cache_free_text": vocab.decode(cf_tokens), "cache_free_ms": cf_ms, "cache_free_calls": cf_calls,
            }
            run["auto_score"] = score(run["auto_text"], utt)
            run["forced_score"] = score(run["forced_text"], utt)
            stt_ms = mel_ms + enc_ms + forced["decode_ms"]
            run["stt_ms"] = stt_ms
            run["stt_rtf"] = (stt_ms / 1e3) / audio_sec if audio_sec else None
            runs.append(run)

        best = min(runs, key=lambda x: x["forced_score"]["cer"])
        rows.append({"utt": utt, "runs": runs, "best": best})
        log(f"{utt['id']:6s} LID {best['auto']['detected']} (p_hi={best['auto']['p_hi']:.2f}) "
            f"CER auto {statistics.fmean(x['auto_score']['cer'] for x in runs):.3f} "
            f"forced {statistics.fmean(x['forced_score']['cer'] for x in runs):.3f} | "
            f"{best['forced_text']}")

    # ---- the app's display pipeline (Kotlin) on every auto-language transcript ----
    apply_display(rows)

    # ---- aggregate ----
    def per_token(run, key):
        if key == "kv":
            n = run["forced"]["decoder_calls"]
            return run["forced"]["decode_ms"] / max(1, n)
        return run["cache_free_ms"] / max(1, run["cache_free_calls"])

    # CER/WER/LID: the original corpus only (comparable with earlier sessions); the held-out
    # sentences (kind "heldout", written before the post-corrector saw any output) separately.
    # Latency statistics use every run.
    held = [row for row in rows if row["utt"].get("kind") == "heldout"]
    corpus = [row for row in rows if row["utt"].get("kind") != "heldout"]
    all_runs = [x for row in rows for x in row["runs"]]
    corpus_runs = [x for row in corpus for x in row["runs"]]
    lid_ok = [x["auto"]["detected"] == row["utt"]["lang"] for row in corpus for x in row["runs"]]
    same_tokens = [x["cache_free_text"] == x["forced_text"] for x in all_runs]
    cf_ms = summarize([x["cache_free_ms"] for x in all_runs])
    kv_ms = summarize([x["forced"]["decode_ms"] for x in all_runs])
    parity = g2p_parity(fixture)

    closed = {
        "label": args.label,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host": {"cpu": platform.processor() or platform.machine(), "cpu_model": cpu_model(),
                 "threads": args.threads, "onnxruntime": ort.__version__, "python": platform.python_version()},
        "measurement_note": ("All latencies are measured on the host CPU above with the shipped INT8 "
                             "graphs and the same ORT session options as the app; they are NOT phone "
                             "numbers. The app's telemetry HUD reports live on-device STT/LLM/TTS/E2E."),
        "runs_per_utterance": args.runs,
        "models": {"encoder": os.path.basename(args.encoder), "tts": os.path.basename(args.tts),
                   "decoder": "vaani_stt_decoder.int8.onnx", "decoder_with_past": os.path.basename(past_path or "")},
        "g2p": {
            "phoneme_coverage_pct": fixture["phoneme_coverage_pct"],
            "phoneme_codepoints_total": fixture["phoneme_codepoints_total"],
            "phoneme_codepoints_dropped": fixture["phoneme_codepoints_dropped"],
            "missing_symbols": fixture["missing_symbols"],
            "espeak_parity": parity,
        },
        "mel_frontend": {
            "filterbank": "vaani_mel_filters.bin (Slaney, 80x201)",
            "cpp_vs_numpy_whisper_reference_max_abs_err": summarize(mel_err),
            "fft_frames_fraction": summarize(fft_frac),
            "cpp_ms": summarize([x["mel_ms"] for x in all_runs]),
            "numpy_reference_ms": summarize([x["mel_numpy_ms"] for x in all_runs]),
        },
        "resampler_22k05_to_16k": {"snr_db_vs_scipy_resample_poly": summarize(resample_snr)},
        "stt": {
            "scope": f"CER/WER/LID over the {len(corpus)}-utterance corpus; held-out set in 'heldout'",
            "lid_two_way_accuracy_pct": round(100.0 * sum(lid_ok) / len(lid_ok), 2),
            "auto_language": {"cer": summarize([x["auto_score"]["cer"] for x in corpus_runs]),
                              "wer": summarize([x["auto_score"]["wer"] for x in corpus_runs])},
            "forced_language": {"cer": summarize([x["forced_score"]["cer"] for x in corpus_runs]),
                                "wer": summarize([x["forced_score"]["wer"] for x in corpus_runs])},
            "raw_cer_by_language": by_lang([(x["forced_score"]["cer"], row["utt"]["lang"])
                                            for row in corpus for x in row["runs"]]),
            **display_blocks(corpus, held),
            "encoder_ms": summarize([x["enc_ms"] for x in all_runs]),
            "lid_step0_ms": summarize([x["auto"]["lid_ms"] for x in all_runs]),
            "decoder": {
                "cache_free_session1_ms": cf_ms,
                "kv_cache_session2_ms": kv_ms,
                "cache_free_ms_per_token": summarize([per_token(x, "cf") for x in all_runs]),
                "kv_cache_ms_per_token": summarize([per_token(x, "kv") for x in all_runs]),
                "speedup_mean": round(cf_ms["mean"] / kv_ms["mean"], 2) if cf_ms and kv_ms else None,
                "kv_vs_cache_free_identical_text_pct": round(100.0 * sum(same_tokens) / len(same_tokens), 2),
                "note": "both paths use identical masks (script calibration + Whisper non-speech + "
                        "specials, suppress_blank, repetition guard); identical text => exact KV cache",
            },
            "stt_total_ms": summarize([x["stt_ms"] for x in all_runs]),
            "stt_rtf": summarize([x["stt_rtf"] for x in all_runs]),
        },
        "tts": {
            "rtf": summarize([x["tts_rtf"] for x in all_runs]),
            "first_sentence_ms": summarize([x["tts_first_audio_ms"] for x in all_runs]),
            "raw_peak_amplitude": summarize([x["raw_peak"] for x in all_runs]),
        },
        "utterances": [{
            "id": row["utt"]["id"], "kind": row["utt"].get("kind"), "lang": row["utt"]["lang"], "text": row["utt"]["text"],
            "acoustic_devanagari": row["utt"]["acoustic_devanagari"],
            "audio_sec": round(row["best"]["audio_sec"], 3),
            "lid_detected": [x["auto"]["detected"] for x in row["runs"]],
            "lid_p_hi": [round(x["auto"]["p_hi"], 3) for x in row["runs"]],
            "transcripts_auto": [x["auto_text"] for x in row["runs"]],
            "transcripts_forced": [x["forced_text"] for x in row["runs"]],
            "cer_forced": [x["forced_score"]["cer"] for x in row["runs"]],
            "wer_forced": [x["forced_score"]["wer"] for x in row["runs"]],
            "cer_auto": [x["auto_score"]["cer"] for x in row["runs"]],
            "transcripts_display": [x.get("display_text") for x in row["runs"]],
            "cer_display": [(x.get("display_score") or {}).get("cer") for x in row["runs"]],
            "tts_rtf": round(statistics.fmean(x["tts_rtf"] for x in row["runs"]), 4),
            "stt_ms": round(statistics.fmean(x["stt_ms"] for x in row["runs"]), 1),
            "tokens": [len(x["forced"]["tokens"]) for x in row["runs"]],
        } for row in rows],
    }

    report = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    report.setdefault("closed_loop_history", {})
    report["closed_loop_history"][args.label] = {
        "generated_at": closed["generated_at"],
        "phoneme_coverage_pct": closed["g2p"]["phoneme_coverage_pct"],
        "espeak_segmental_match_pct": (parity or {}).get("segmental_match_pct"),
        "forced_cer_mean": closed["stt"]["forced_language"]["cer"]["mean"],
        "auto_cer_mean": closed["stt"]["auto_language"]["cer"]["mean"],
        "lid_accuracy_pct": closed["stt"]["lid_two_way_accuracy_pct"],
        "raw_cer_hi": (closed["stt"]["raw_cer_by_language"]["hi"] or {}).get("mean"),
        **history_display(closed),
        "encoder": closed["models"]["encoder"],
        "runs_per_utterance": args.runs,
    }
    report["closed_loop"] = closed
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, args.out)                      # atomic: never leave a half-written report
    log(f"wrote {args.out}")
    log(json.dumps({k: closed["stt"][k] for k in ("lid_two_way_accuracy_pct", "auto_language", "forced_language",
                                                  "raw_cer_by_language")}, ensure_ascii=False))
    log("display:", json.dumps(closed["stt"]["display_transcript"], ensure_ascii=False))
    log("post-corrector:", json.dumps(closed["stt"]["post_corrector"], ensure_ascii=False))
    log("heldout:", json.dumps({k: v for k, v in (closed["stt"]["heldout"] or {}).items() if k != "samples"},
                               ensure_ascii=False))
    for row in rows:
        log(f"  {row['utt']['id']:6s} display: {row['runs'][0].get('display_text')}")
    log("decoder:", json.dumps(closed["stt"]["decoder"], ensure_ascii=False))
    log("mel:", json.dumps(closed["mel_frontend"], ensure_ascii=False))
    log("resampler:", json.dumps(closed["resampler_22k05_to_16k"]), "tts:", json.dumps(closed["tts"]))
    if parity:
        log("espeak parity:", {k: parity[k] for k in ("words", "exact_match_pct", "segmental_match_pct",
                                                      "phoneme_error_rate_pct")})


def apply_display(rows):
    """Attaches the app's display transcript (Kotlin) and its scores to every run."""
    disp = kotlin_display([{"key": f"{row['utt']['id']}|{i}", "raw": x["auto_text"], "lang": x["auto"]["lang"]}
                           for row in rows for i, x in enumerate(row["runs"])]) or {}
    for row in rows:
        for i, x in enumerate(row["runs"]):
            d = disp.get(f"{row['utt']['id']}|{i}")
            x["display_text"] = d["display"] if d else None
            x["display_us"] = d["us"] if d else None
            x["display_score"] = score_display(d["display"], row["utt"]) if d else None
    return bool(disp)


def display_blocks(corpus, held):
    runs = [x for row in corpus + held for x in row["runs"]]
    return {
        "display_transcript": {
            "what": ("HinglishCodec.displayTranscript(raw, detected language): the text the app shows and "
                     "sends to the LLM, scored in Roman script against the reference text"),
            "cer_by_language": by_lang([(x["display_score"]["cer"], row["utt"]["lang"])
                                        for row in corpus for x in row["runs"] if x.get("display_score")]),
            "wer_by_language": by_lang([(x["display_score"]["wer"], row["utt"]["lang"])
                                        for row in corpus for x in row["runs"] if x.get("display_score")]),
            "normalizer_us_per_call": summarize([x["display_us"] for x in runs if x.get("display_us") is not None]),
        },
        "post_corrector": {
            "what": ("paired raw -> display CER on the same transcripts (auto language, Latin-script raw "
                     "output only, so both sides are scored against the same Roman reference)"),
            "corpus": post_corrector_effect(corpus),
            "heldout": post_corrector_effect(held),
        },
        "heldout": heldout_block(held),
    }


def history_display(closed):
    st = closed["stt"]
    return {
        "display_cer_mean": (st["display_transcript"]["cer_by_language"]["all"] or {}).get("mean"),
        "display_cer_hi": (st["display_transcript"]["cer_by_language"]["hi"] or {}).get("mean"),
        "heldout_raw_cer_mean": ((st["heldout"] or {}).get("raw_auto_cer", {}).get("all") or {}).get("mean"),
        "heldout_display_cer_mean": ((st["heldout"] or {}).get("display_cer", {}).get("all") or {}).get("mean"),
    }


def rescore_display(args):
    """Display pass only, on the stored transcripts of the last full run (same seeded audio and
    decoding), e.g. after a post-corrector change: no TTS/STT re-run needed."""
    report = json.load(open(args.out, encoding="utf-8"))
    closed = report["closed_loop"]
    rows = []
    for u in closed["utterances"]:
        runs = [{"auto_text": raw, "auto": {"detected": lid, "lang": lid},
                 "auto_score": {"cer": cer, "space": "devanagari" if is_devanagari(raw) else "latin"}}
                for raw, lid, cer in zip(u["transcripts_auto"], u["lid_detected"], u["cer_auto"])]
        rows.append({"utt": {"id": u["id"], "kind": u.get("kind"), "lang": u["lang"], "text": u["text"]}, "runs": runs})
    if not apply_display(rows):
        sys.exit("display pass failed")
    for row, u in zip(rows, closed["utterances"]):
        u["transcripts_display"] = [x["display_text"] for x in row["runs"]]
        u["cer_display"] = [x["display_score"]["cer"] for x in row["runs"]]
    held = [row for row in rows if row["utt"].get("kind") == "heldout"]
    corpus = [row for row in rows if row["utt"].get("kind") != "heldout"]
    closed["stt"].update(display_blocks(corpus, held))
    closed["display_rescored_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    report["closed_loop_history"].setdefault(closed["label"], {}).update(history_display(closed))
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, args.out)
    log(f"rescored display transcripts -> {args.out}")
    log("post-corrector:", json.dumps(closed["stt"]["post_corrector"], ensure_ascii=False))


def paired_change(pairs):
    """[(before, after)] -> mean(after - before) with a 95% bootstrap CI."""
    d = np.array([b - a for a, b in pairs], dtype=np.float64)
    if d.size == 0:
        return None
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, d.size, size=(2000, d.size))].mean(axis=1)
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                                       round(float(np.percentile(boots, 97.5)), 4)],
            "pairs": int(d.size), "improved": int((d < 0).sum()), "worsened": int((d > 0).sum())}


def post_corrector_effect(rows):
    """Raw (auto) vs display CER per language on the same transcripts."""
    out = {}
    for lang in ("en", "hi", "all"):
        pairs = [(x["auto_score"]["cer"], x["display_score"]["cer"]) for row in rows for x in row["runs"]
                 if (lang == "all" or row["utt"]["lang"] == lang) and x.get("display_score")
                 and x["auto_score"]["space"] == "latin"]
        if pairs:
            out[lang] = {"raw_cer": round(statistics.fmean(a for a, _ in pairs), 4),
                         "display_cer": round(statistics.fmean(b for _, b in pairs), 4),
                         "change": paired_change(pairs)}
    out["devanagari_raw_outputs_excluded"] = sum(1 for row in rows for x in row["runs"] if x["auto_score"]["space"] != "latin")
    return out


def heldout_block(rows):
    if not rows:
        return None
    runs = [(x, row) for row in rows for x in row["runs"]]
    return {
        "utterances": [row["utt"]["id"] for row in rows],
        "lid_two_way_accuracy_pct": round(100.0 * statistics.fmean(x["auto"]["detected"] == r["utt"]["lang"] for x, r in runs), 2),
        "raw_auto_cer": by_lang([(x["auto_score"]["cer"], r["utt"]["lang"]) for x, r in runs]),
        "display_cer": by_lang([(x["display_score"]["cer"], r["utt"]["lang"]) for x, r in runs if x.get("display_score")]),
        "display_wer": by_lang([(x["display_score"]["wer"], r["utt"]["lang"]) for x, r in runs if x.get("display_score")]),
        "samples": {row["utt"]["id"]: {"ref": row["utt"]["text"], "raw": row["runs"][0]["auto_text"],
                                       "display": row["runs"][0].get("display_text")} for row in rows},
    }


def paired_delta(rows, base, other, key):
    """Mean paired CER difference (other - base) over identical audio, 95% bootstrap CI."""
    a = {(x["utt"], x["run"]): x[key] for x in rows if x["enc"] == base and x.get(key) is not None}
    b = {(x["utt"], x["run"]): x[key] for x in rows if x["enc"] == other and x.get(key) is not None}
    d = np.array([b[k] - a[k] for k in a if k in b], dtype=np.float64)
    if d.size == 0:
        return None
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, d.size, size=(2000, d.size))].mean(axis=1)
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
                                                       round(float(np.percentile(boots, 97.5)), 4)],
            "pairs": int(d.size), "better": int((d < 0).sum()), "worse": int((d > 0).sum())}


def run_encoder_ab(args, fixture, filters, dsp, stt, tts):
    """Paired encoder precision study: each encoder transcribes the same seeded TTS audio."""
    encs = []
    for spec in args.ab_encoders.split(","):
        name, path = spec.split("=", 1)
        sess = make_session(path, args.threads)
        zeros = ort.OrtValue.ortvalue_from_numpy(np.zeros((1, N_MELS, N_FRAMES), np.float32))
        sess.run_with_ort_values(["last_hidden_state"], {"input_features": zeros})           # warm-up
        encs.append((name, path, sess))
    rows = []
    for utt in fixture["utterances"]:
        for r in range(args.runs):
            audio22, _, _, _ = synthesize(tts, utt)
            audio_sec = len(audio22) / TTS_SR
            audio16 = dsp.resample(audio22, TTS_SR, STT_SR) if dsp else resample_poly(audio22, 320, 441).astype(np.float32)
            mel = dsp.log_mel(audio16, filters)[0] if dsp else whisper_log_mel_reference(audio16, filters)
            feed = {"input_features": ort.OrtValue.ortvalue_from_numpy(mel)}
            for name, _, sess in encs:
                t0 = time.perf_counter()
                hidden = sess.run_with_ort_values(["last_hidden_state"], feed)[0]
                enc_ms = (time.perf_counter() - t0) * 1e3
                auto = stt.decode_kv(hidden, "auto", audio_sec)
                forced = auto if auto["lang"] == utt["lang"] else stt.decode_kv(hidden, utt["lang"], audio_sec)
                forced_text = stt.vocab.decode(forced["tokens"])
                rows.append({"enc": name, "utt": utt["id"], "lang": utt["lang"], "run": r, "enc_ms": enc_ms,
                             "lid_ok": auto["detected"] == utt["lang"], "auto_lang": auto["lang"],
                             "auto_text": stt.vocab.decode(auto["tokens"]), "forced_text": forced_text,
                             "raw_cer": score(forced_text, utt)["cer"]})
        line = " | ".join(f"{n} {statistics.fmean(x['raw_cer'] for x in rows if x['enc'] == n and x['utt'] == utt['id']):.3f}"
                          for n, _, _ in encs)
        log(f"{utt['id']:6s} raw CER  {line}")

    disp = kotlin_display([{"key": f"{x['enc']}|{x['utt']}|{x['run']}", "raw": x["auto_text"], "lang": x["auto_lang"]}
                           for x in rows]) or {}
    utts = {u["id"]: u for u in fixture["utterances"]}
    for x in rows:
        d = disp.get(f"{x['enc']}|{x['utt']}|{x['run']}")
        x["display_text"] = d["display"] if d else None
        x["display_cer"] = score_display(d["display"], utts[x["utt"]])["cer"] if d else None

    base = encs[0][0]
    summary = {}
    for name, path, _ in encs:
        rs = [x for x in rows if x["enc"] == name]
        summary[name] = {
            "file": os.path.basename(path), "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "encoder_ms_host": summarize([x["enc_ms"] for x in rs]),
            "lid_accuracy_pct": round(100.0 * sum(x["lid_ok"] for x in rs) / len(rs), 2),
            "raw_cer_forced_language": by_lang([(x["raw_cer"], x["lang"]) for x in rs]),
            "display_cer": by_lang([(x["display_cer"], x["lang"]) for x in rs if x["display_cer"] is not None]),
        }
        if name != base:
            summary[name]["paired_vs_" + base] = {
                "raw_cer_hi": paired_delta([x for x in rows if x["lang"] == "hi"], base, name, "raw_cer"),
                "raw_cer_en": paired_delta([x for x in rows if x["lang"] == "en"], base, name, "raw_cer"),
                "display_cer_all": paired_delta(rows, base, name, "display_cer"),
            }
        log(f"{name:8s} raw CER {json.dumps(summary[name]['raw_cer_forced_language'])} "
            f"display CER {json.dumps(summary[name]['display_cer'])} enc {summary[name]['encoder_ms_host']}")
    study = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "method": (f"paired: {len(fixture['utterances'])} utterances x {args.runs} seeded VITS samples (seed {args.seed}); "
                   "every encoder transcribes the identical audio/mel; same INT8 decoders + KV cache; "
                   "paired deltas = mean(other - base) with 95% bootstrap CI over utterance-samples; "
                   "encoder_ms on this host with onnxruntime " + ort.__version__),
        "host": {"cpu_model": cpu_model(), "threads": args.threads},
        "baseline": base,
        "encoders": summary,
        "samples_first_run": {name: {x["utt"]: {"raw": x["forced_text"], "display": x["display_text"]}
                                     for x in rows if x["enc"] == name and x["run"] == 0}
                              for name, _, _ in encs},
    }
    report = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
    report["encoder_precision_ab"] = study
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, args.out)
    log(f"wrote {args.out} (encoder_precision_ab)")
    for name in summary:
        for k, v in summary[name].items():
            if k.startswith("paired_vs_"):
                log(f"{name} {k}: {json.dumps(v)}")


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


if __name__ == "__main__":
    main()
