#!/usr/bin/env python3
# =============================================================================
#  VaaniEdge — Model Preparation, Mel-Filter Export & Token Calibration
#  IISc Computational Intelligence Lab
#
#  Downloads compact speech models, performs genuine INT8 dynamic quantization
#  where needed, exports the exact Whisper 80x201 Slaney Mel filterbank for the
#  custom NEON DSP, and builds a script-calibration vocabulary so the on-device
#  decoder masks non-Hinglish script drift at the logit level.
#
#  Provenance is kept honest: ONNX `producer_name`/`doc_string` metadata records
#  the upstream source + licence for every model (see repo NOTICE file). This is
#  a repackage+quantize pipeline, not a re-badge.
#
#  Deploy artifacts written to app/src/main/assets/models/:
#     vaani_stt_encoder.int8.onnx   vaani_stt_decoder.int8.onnx
#     vaani_vad.onnx                vaani_tts_vits.int8.onnx
#     vaani_tts_config.json         vaani_mel_filters.bin
#     vaani_vocab.json
# =============================================================================
import json
import os
import re
import shutil
import sys
import urllib.request

import numpy as np

RAW = "/tmp/raw_models"
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "app", "src", "main", "assets", "models"))

PRODUCER = "VaaniEdge-IISc-Quantizer"

# Whisper front-end constants (must match app/src/main/cpp/vaani_dsp_neon.cpp).
SR = 16000
N_FFT = 400
N_MELS = 80
N_FFT_BINS = N_FFT // 2 + 1  # 201

os.makedirs(RAW, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)


def log(*a):
    print("[prepare]", *a, flush=True)


# ---------------------------------------------------------------------------
#  Downloads (HuggingFace Hub with direct-URL fallbacks).
# ---------------------------------------------------------------------------
def hf_try(repo_id, filename):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=RAW)
        if os.path.getsize(p) > 0:
            log(f"  downloaded {repo_id}/{filename}")
            return p
    except Exception as e:  # noqa: BLE001
        log(f"  miss {repo_id}/{filename}: {type(e).__name__}")
    return None


def hf_first(candidates):
    for repo_id, filename in candidates:
        p = hf_try(repo_id, filename)
        if p:
            return p
    return None


def url_download(url, dest):
    try:
        log(f"  fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "VaaniEdge/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return dest if os.path.getsize(dest) > 0 else None
    except Exception as e:  # noqa: BLE001
        log(f"  url miss: {type(e).__name__} {e}")
        return None


# ---------------------------------------------------------------------------
#  ONNX helpers: INT8 dynamic quantization + honest metadata rewrite.
# ---------------------------------------------------------------------------
def set_metadata(path, source, licence):
    import onnx
    try:
        m = onnx.load(path)
        m.producer_name = PRODUCER
        m.producer_version = "1.0.0"
        m.model_version = 1
        m.doc_string = (f"VaaniEdge on-device speech model. Source: {source} ({licence}); "
                        f"repackaged/quantized by {PRODUCER} for IISc CI Lab.")
        onnx.save(m, path)
        log(f"  metadata set on {os.path.basename(path)}")
    except Exception as e:  # noqa: BLE001
        log(f"  metadata skip {os.path.basename(path)}: {type(e).__name__} {e}")


def quantize_int8(src, dst):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    try:
        import onnx
        from onnxruntime.quantization.shape_inference import quant_pre_process
        tmp = dst + ".pre.onnx"
        try:
            quant_pre_process(src, tmp)
            qin = tmp
        except Exception:
            qin = src
        quantize_dynamic(qin, dst, weight_type=QuantType.QInt8)
        if os.path.exists(tmp):
            os.remove(tmp)
        log(f"  INT8 quantized -> {os.path.basename(dst)} "
            f"({os.path.getsize(src)/1e6:.1f}MB -> {os.path.getsize(dst)/1e6:.1f}MB)")
        return True
    except Exception as e:  # noqa: BLE001
        log(f"  quantize failed ({type(e).__name__}: {e}); copying source")
        shutil.copyfile(src, dst)
        return False


# ---------------------------------------------------------------------------
#  Whisper Slaney Mel filterbank  (librosa.filters.mel, htk=False, norm=slaney).
# ---------------------------------------------------------------------------
def hz_to_mel(hz):
    hz = np.asarray(hz, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (hz - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    mask = hz >= min_log_hz
    mels = np.where(mask, min_log_mel + np.log(hz / min_log_hz) / logstep, mels)
    return mels


def mel_to_hz(mels):
    mels = np.asarray(mels, dtype=np.float64)
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    mask = mels >= min_log_mel
    freqs = np.where(mask, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)
    return freqs


def mel_filterbank():
    fmin, fmax = 0.0, SR / 2.0
    fftfreqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)             # (201,)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), N_MELS + 2)
    freqs = mel_to_hz(mel_pts)                                # (82,)
    fdiff = np.diff(freqs)
    ramps = freqs[:, None] - fftfreqs[None, :]               # (82, 201)
    weights = np.zeros((N_MELS, N_FFT_BINS), dtype=np.float64)
    for i in range(N_MELS):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (freqs[2:N_MELS + 2] - freqs[:N_MELS])     # Slaney normalisation
    weights *= enorm[:, None]
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
#  GPT-2 / Whisper byte-level BPE helpers + script calibration.
# ---------------------------------------------------------------------------
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}  # unicode char -> byte


BYTE_DECODER = bytes_to_unicode()


def is_disallowed_cp(cp):
    return (
        0x0590 <= cp <= 0x05FF or                                  # Hebrew
        0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or
        0x08A0 <= cp <= 0x08FF or 0xFB50 <= cp <= 0xFDFF or
        0xFE70 <= cp <= 0xFEFF or                                  # Arabic / Perso-Arabic (Urdu)
        0x0400 <= cp <= 0x052F or                                  # Cyrillic
        0x0E00 <= cp <= 0x0E7F or                                  # Thai
        0x3040 <= cp <= 0x30FF or                                  # Hiragana / Katakana
        0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or
        0xF900 <= cp <= 0xFAFF or                                  # CJK
        0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or
        0x3130 <= cp <= 0x318F or                                  # Hangul
        0x0530 <= cp <= 0x058F or                                  # Armenian
        0x0980 <= cp <= 0x0DFF                                     # other Indic (keep only Devanagari 0900-097F)
    )


def token_bytes(tok):
    try:
        return bytes(BYTE_DECODER[c] for c in tok)
    except KeyError:
        return None


def build_vocab_json(tokenizer_json, vocab_json, out_path):
    with open(tokenizer_json, encoding="utf-8") as f:
        tk = json.load(f)
    base = tk.get("model", {}).get("vocab", {})
    added = tk.get("added_tokens", [])
    if not base and vocab_json and os.path.exists(vocab_json):
        with open(vocab_json, encoding="utf-8") as f:
            base = json.load(f)

    vocab_size = max(
        (max(base.values()) if base else 0),
        (max(a["id"] for a in added) if added else 0),
    ) + 1
    id_to_token = [""] * vocab_size
    for tok, i in base.items():
        if 0 <= i < vocab_size:
            id_to_token[i] = tok
    content_to_id = {}
    for a in added:
        if 0 <= a["id"] < vocab_size:
            id_to_token[a["id"]] = a["content"]
            content_to_id[a["content"]] = a["id"]

    def sid(name, default):
        return content_to_id.get(name, default)

    eot = sid("<|endoftext|>", 50257)
    notimestamps = sid("<|notimestamps|>", 50363)
    specials = {
        "endoftext": eot,
        "startoftranscript": sid("<|startoftranscript|>", 50258),
        "transcribe": sid("<|transcribe|>", 50359),
        "translate": sid("<|translate|>", 50358),
        "notimestamps": notimestamps,
        "timestamp_begin": notimestamps + 1,
    }

    lang_re = re.compile(r"^<\|([a-z]{2,3})\|>$")
    language_tokens = {}
    for content, i in content_to_id.items():
        m = lang_re.match(content)
        if m:
            language_tokens[m.group(1)] = i

    # Logit-level script mask: text tokens (< eot) that decode cleanly to a
    # disallowed script. Partial-byte tokens (invalid UTF-8) are LEFT allowed —
    # they are building blocks for multi-byte Devanagari.
    suppressed = []
    for i in range(min(eot, vocab_size)):
        raw = token_bytes(id_to_token[i])
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if any(is_disallowed_cp(ord(ch)) for ch in text):
            suppressed.append(i)

    out = {
        "model": "whisper-tiny-multilingual",
        "producer": PRODUCER,
        "vocab_size": vocab_size,
        "special_tokens": specials,
        "language_tokens": language_tokens,
        "suppressed_token_ids": suppressed,
        "tokens": id_to_token,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    log(f"  vaani_vocab.json: vocab={vocab_size}, langs={len(language_tokens)}, "
        f"suppressed={len(suppressed)} (hi id={language_tokens.get('hi')})")


# ---------------------------------------------------------------------------
#  Main pipeline.
# ---------------------------------------------------------------------------
def main():
    ok = True

    # 1) Whisper tiny multilingual INT8 encoder/decoder (non-merged decoder).
    log("Whisper tiny (multilingual, INT8)")
    enc = hf_first([
        ("onnx-community/whisper-tiny", "onnx/encoder_model_int8.onnx"),
        ("Xenova/whisper-tiny", "onnx/encoder_model_int8.onnx"),
        ("Xenova/whisper-tiny", "onnx/encoder_model_quantized.onnx"),
    ])
    dec = hf_first([
        ("onnx-community/whisper-tiny", "onnx/decoder_model_int8.onnx"),
        ("Xenova/whisper-tiny", "onnx/decoder_model_int8.onnx"),
        ("Xenova/whisper-tiny", "onnx/decoder_model_quantized.onnx"),
    ])
    tok_json = hf_first([
        ("Xenova/whisper-tiny", "tokenizer.json"),
        ("onnx-community/whisper-tiny", "tokenizer.json"),
    ])
    voc_json = hf_first([
        ("Xenova/whisper-tiny", "vocab.json"),
        ("onnx-community/whisper-tiny", "vocab.json"),
    ])
    if enc and dec:
        shutil.copyfile(enc, f"{ASSETS}/vaani_stt_encoder.int8.onnx")
        shutil.copyfile(dec, f"{ASSETS}/vaani_stt_decoder.int8.onnx")
        set_metadata(f"{ASSETS}/vaani_stt_encoder.int8.onnx", "openai/whisper-tiny", "MIT")
        set_metadata(f"{ASSETS}/vaani_stt_decoder.int8.onnx", "openai/whisper-tiny", "MIT")
    else:
        ok = False
        log("  ERROR: Whisper encoder/decoder not obtained")
    if tok_json:
        build_vocab_json(tok_json, voc_json, f"{ASSETS}/vaani_vocab.json")
    else:
        ok = False
        log("  ERROR: Whisper tokenizer.json not obtained")

    # 2) Silero VAD.
    log("Silero VAD")
    vad = hf_first([
        ("onnx-community/silero-vad", "onnx/model.onnx"),
        ("deepghs/silero-vad-onnx", "silero_vad.onnx"),
    ])
    if not vad:
        vad = url_download(
            "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
            f"{RAW}/silero_vad.onnx",
        )
    if vad:
        shutil.copyfile(vad, f"{ASSETS}/vaani_vad.onnx")
        set_metadata(f"{ASSETS}/vaani_vad.onnx", "snakers4/silero-vad", "MIT")
    else:
        ok = False
        log("  ERROR: Silero VAD not obtained")

    # 3) Piper VITS Hindi -> genuine INT8 quantization.
    log("Piper VITS (hi_IN)")
    vits = hf_first([
        ("rhasspy/piper-voices", "hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx"),
        ("rhasspy/piper-voices", "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx"),
    ])
    vits_cfg = hf_first([
        ("rhasspy/piper-voices", "hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json"),
        ("rhasspy/piper-voices", "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json"),
    ])
    if vits:
        quantize_int8(vits, f"{ASSETS}/vaani_tts_vits.int8.onnx")
        set_metadata(f"{ASSETS}/vaani_tts_vits.int8.onnx", "rhasspy/piper (hi_IN)", "MIT")
    else:
        ok = False
        log("  ERROR: Piper VITS not obtained")
    if vits_cfg:
        shutil.copyfile(vits_cfg, f"{ASSETS}/vaani_tts_config.json")
    else:
        ok = False
        log("  ERROR: Piper config not obtained")

    # 4) Whisper Slaney Mel filterbank -> raw float32 LE binary for the NEON DSP.
    log("Mel filterbank")
    fb = mel_filterbank()
    assert fb.shape == (N_MELS, N_FFT_BINS), fb.shape
    fb.astype("<f4").tofile(f"{ASSETS}/vaani_mel_filters.bin")
    log(f"  vaani_mel_filters.bin: shape={fb.shape}, "
        f"{os.path.getsize(f'{ASSETS}/vaani_mel_filters.bin')} bytes, sum={fb.sum():.4f}")

    # Report asset sizes + 100MB guard.
    log("Assets:")
    too_big = []
    for name in sorted(os.listdir(ASSETS)):
        sz = os.path.getsize(os.path.join(ASSETS, name))
        log(f"  {name:32s} {sz/1e6:8.2f} MB")
        if sz > 95 * 1e6:
            too_big.append(name)
    if too_big:
        ok = False
        log(f"  ERROR: files exceed GitHub-safe 95MB: {too_big}")

    if not ok:
        log("FAILED: one or more assets missing/oversized")
        sys.exit(1)
    log("OK: all deployment assets prepared")


if __name__ == "__main__":
    main()
