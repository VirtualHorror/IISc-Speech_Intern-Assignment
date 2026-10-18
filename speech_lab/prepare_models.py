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
#     vaani_stt_decoder_with_past.int8.onnx   (KV-cache decoder, Session 2)
#     vaani_vad.onnx                vaani_tts_vits.fp16.onnx
#
#  Runtime contract: every graph must load on the APK's ONNX Runtime (Android
#  1.18.0). ORT <= 1.22 has no ConvInteger kernel for int8 weights, so the encoder
#  is re-quantized MatMul-only (its two front-end convs stay fp32) and the VITS voice
#  ships fp16-stored / fp32-computed (INT8 ConvInteger VITS was also 3.2x SLOWER on
#  CPU). speech_lab/check_ort_compat.py verifies this under onnxruntime==1.18.0.
#     vaani_tts_config.json         vaani_mel_filters.bin
#     vaani_vocab.json
#  and app/src/main/assets/lexicon/vaani_en_cmudict.txt (English pronunciations).
# =============================================================================
import argparse
import json
import os
import re
import shutil
import subprocess
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


def fp16_storage(src, dst, min_elems=16):
    """fp32 graph -> fp16-stored weights + Cast(to=float) nodes. ORT constant-folds the
    Casts at session creation, so inference runs at full fp32 speed/accuracy while the
    file (and APK) is half the size."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(src)
    g = m.graph
    casts, keep = [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elems:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            keep.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], name=init.name + "__cast",
                                          to=TensorProto.FLOAT))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.save(m, dst)
    log(f"  fp16-storage -> {os.path.basename(dst)} ({os.path.getsize(src)/1e6:.1f}MB -> "
        f"{os.path.getsize(dst)/1e6:.1f}MB, {len(casts)} weight tensors)")


def quantize_matmul_int8(src, dst):
    """Dynamic INT8 for MatMul only (DynamicQuantizeLinear + MatMulInteger, both supported
    by ORT 1.18 for int8 weights); Conv stays fp32 because ConvInteger(int8) is not."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    tmp = dst + ".pre.onnx"
    quant_pre_process(src, tmp)
    quantize_dynamic(tmp, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    os.remove(tmp)
    log(f"  MatMul-only INT8 -> {os.path.basename(dst)} "
        f"({os.path.getsize(src)/1e6:.1f}MB -> {os.path.getsize(dst)/1e6:.1f}MB)")


def quantize_decoder_shared_table(src, dst):
    """INT8 Whisper decoder from an optimum fp32 export, one shared token table.

    The exporter stores the tied output projection as a second, pre-transposed copy of the
    51865 x d embedding ([d, 51865], identical values), doubling the file. MatMul-only
    dynamic INT8 quantises that copy for the logits; the token-embedding Gather is then
    re-pointed at the same INT8 table (Gather(axis=1) + Transpose + DequantizeLinear), so
    the fp32 embedding is dropped and no table is de-quantised per step.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic

    tmp = dst + ".pre.onnx"
    quantize_dynamic(src, tmp, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
    m = onnx.load(tmp)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    emb_name = "model.decoder.embed_tokens.weight"
    gather = next(n for n in g.node if n.op_type == "Gather" and n.input[0] == emb_name)
    emb = numpy_helper.to_array(inits[emb_name])                                   # [V, d] fp32
    # the quantised proj_out weight: an int8 [d, V] initializer feeding a MatMulInteger
    cand = [i for i in g.initializer if list(i.dims) == [emb.shape[1], emb.shape[0]] and i.name.endswith("_quantized")]
    assert len(cand) == 1, [c.name for c in cand]
    wq = cand[0]
    mmi = next(n for n in g.node if n.op_type in ("MatMulInteger", "DynamicQuantizeMatMul") and wq.name in n.input)
    if mmi.op_type == "MatMulInteger":
        zp_name = mmi.input[3]
        scale_name = wq.name.replace("_quantized", "_scale")
    else:
        scale_name, zp_name = mmi.input[2], mmi.input[3]
    assert scale_name in inits and zp_name in inits, (scale_name, zp_name)
    deq = numpy_helper.to_array(wq).astype(np.float32).T
    deq = (deq - numpy_helper.to_array(inits[zp_name]).astype(np.float32)) * numpy_helper.to_array(inits[scale_name])
    err = float(np.max(np.abs(deq - emb)))
    out = gather.output[0]
    new_nodes = []
    for n in g.node:
        if n is gather:
            new_nodes.append(helper.make_node("Gather", [wq.name, n.input[1]], [out + "_q_axis1"], name=n.name, axis=1))
            new_nodes.append(helper.make_node("Transpose", [out + "_q_axis1"], [out + "_q"], name=n.name + "_to_bld",
                                              perm=[1, 2, 0]))
            new_nodes.append(helper.make_node("DequantizeLinear", [out + "_q", scale_name, zp_name], [out],
                                              name=n.name + "_dequant"))
            continue
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.remove(inits[emb_name])
    onnx.save(m, dst)
    os.remove(tmp)
    log(f"  INT8 decoder (shared table, max |dequant - fp32 emb| = {err:.4f}) -> {os.path.basename(dst)} "
        f"({os.path.getsize(src)/1e6:.1f}MB -> {os.path.getsize(dst)/1e6:.1f}MB)")


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


def fuse_tied_logits_int8(path):
    """Rewrites the tied-embedding output projection of an INT8 Whisper decoder.

    The community INT8 export computes logits as
        MatMul(h, DequantizeLinear(Transpose(W_q)))      W_q: uint8 [51865, 384]
    and ORT never constant-folds DequantizeLinear, so EVERY decoding step
    de-quantises the whole 51865x384 table to fp32 (~80 MB) before a float GEMV:
    ~75% of a KV-cached step. We replace it with the same INT8 kernel every other
    projection in the graph already uses:
        logits = com.microsoft.DynamicQuantizeMatMul(h, W_qT, scale, zero_point)
    and re-point the token-embedding Gather at that single transposed table
    (Gather(axis=1) + Transpose), so the file does not grow.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    wq_name = "model.decoder.embed_tokens.weight_quantized"
    if wq_name not in inits:
        log(f"  tied-logits fusion: {os.path.basename(path)} already rewritten / unknown layout")
        return False
    prod = {o: n for n in g.node for o in n.output}
    tr = next(n for n in g.node if n.op_type == "Transpose" and n.input[0] == wq_name)
    dq = next(n for n in g.node if n.op_type == "DequantizeLinear" and n.input[0] == tr.output[0])
    mm = next(n for n in g.node if n.op_type == "MatMul" and dq.output[0] in n.input)
    gather = next(n for n in g.node if n.op_type == "Gather" and n.input[0] == wq_name)
    scale, zp = dq.input[1], dq.input[2]
    act = mm.input[0] if mm.input[1] == dq.output[0] else mm.input[1]

    wt = np.ascontiguousarray(numpy_helper.to_array(inits[wq_name]).T)        # [384, 51865]
    wt_name = "model.decoder.embed_tokens.weight_t_quantized"
    g.initializer.remove(inits[wq_name])
    g.initializer.append(numpy_helper.from_array(wt, wt_name))

    gather_out = gather.output[0]
    tmp = gather_out + "_axis1"
    new_nodes = []
    for n in g.node:
        if n is tr or n is dq:
            continue
        if n is gather:
            new_nodes.append(helper.make_node("Gather", [wt_name, n.input[1]], [tmp], name=n.name, axis=1))
            new_nodes.append(helper.make_node("Transpose", [tmp], [gather_out], name=n.name + "_to_bld",
                                              perm=[1, 2, 0]))
            continue
        if n is mm:
            new_nodes.append(helper.make_node("DynamicQuantizeMatMul", [act, wt_name, scale, zp],
                                              list(mm.output), name="/proj_out/DynamicQuantizeMatMul",
                                              domain="com.microsoft"))
            continue
        new_nodes.append(n)
    del g.node[:]
    g.node.extend(new_nodes)
    if not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save(m, path)
    log(f"  tied-logits fusion: {os.path.basename(path)} -> DynamicQuantizeMatMul (no per-step table dequant)")
    return True


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


def whisper_non_speech_tokens(tokenizer_json):
    """Exact port of whisper.tokenizer.Tokenizer.non_speech_tokens: speaker tags, music
    notes and bracketed annotations that Whisper suppresses during decoding."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        log("  (tokenizers not installed: non_speech_token_ids left empty)")
        return []
    tk = Tokenizer.from_file(tokenizer_json)
    enc = lambda t: tk.encode(t, add_special_tokens=False).ids  # noqa: E731
    symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
    symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
    miscellaneous = set("♩♪♫♬♭♮♯")
    result = {enc(" -")[0], enc(" '")[0]}
    for symbol in symbols + list(miscellaneous):
        for tokens in (enc(symbol), enc(" " + symbol)):
            if len(tokens) == 1 or symbol in miscellaneous:
                result.add(tokens[0])
    return sorted(result)


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

    non_speech = whisper_non_speech_tokens(tokenizer_json)
    blank = base.get("\u0120", 220)   # "Ġ" = a bare space, suppressed on the first step

    out = {
        "model": "whisper-tiny-multilingual",
        "producer": PRODUCER,
        "vocab_size": vocab_size,
        "special_tokens": specials,
        "language_tokens": language_tokens,
        "suppressed_token_ids": suppressed,
        "non_speech_token_ids": non_speech,
        "blank_token_id": blank,
        "tokens": id_to_token,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    log(f"  vaani_vocab.json: vocab={vocab_size}, langs={len(language_tokens)}, "
        f"suppressed={len(suppressed)}, non_speech={len(non_speech)} "
        f"(hi id={language_tokens.get('hi')}, en id={language_tokens.get('en')})")


# ---------------------------------------------------------------------------
#  Main pipeline.
# ---------------------------------------------------------------------------
CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
CMUDICT_LICENSE_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/LICENSE"
LEXICON_DIR = os.path.abspath(os.path.join(ASSETS, "..", "lexicon"))


def prepare_lexicon():
    """CMU Pronouncing Dictionary -> sorted 'word ARPABET...' asset for the on-device
    English->Devanagari path of HinglishCodec (BSD-style licence kept in the header)."""
    log("CMU Pronouncing Dictionary (English lexicon)")
    raw = url_download(CMUDICT_URL, f"{RAW}/cmudict.dict")
    lic = url_download(CMUDICT_LICENSE_URL, f"{RAW}/cmudict.LICENSE")
    if not raw or not lic:
        log("  ERROR: cmudict not obtained")
        return False
    entries = {}
    word_re = re.compile(r"^[a-z][a-z']{0,19}$")
    with open(raw, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith(";;;"):
                continue
            word, _, phones = line.partition(" ")
            if "(" in word or not word_re.match(word) or word in entries:
                continue                                  # first pronunciation only
            entries[word] = " ".join(phones.split())
    os.makedirs(LEXICON_DIR, exist_ok=True)
    out = os.path.join(LEXICON_DIR, "vaani_en_cmudict.txt")
    with open(lic, encoding="utf-8") as f:
        licence = [";;; " + l.rstrip() for l in f]
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(";;; VaaniEdge English lexicon: first pronunciation of each [a-z'] word from the\n")
        f.write(";;; CMU Pronouncing Dictionary (github.com/cmusphinx/cmudict), sorted for binary search.\n")
        f.write("\n".join(licence) + "\n")
        for w in sorted(entries):
            f.write(f"{w} {entries[w]}\n")
    log(f"  vaani_en_cmudict.txt: {len(entries)} words, {os.path.getsize(out) / 1e6:.2f} MB")
    return True


STEPS = ("stt", "stt-hi", "vocab", "vad", "tts", "mel", "lexicon")

# Hindi specialist for Hindi turns, chosen with speech_lab/eval_hindi_asr.py (FLEURS hi + the
# closed-loop commands). Exporting it needs torch + optimum (a separate venv): pass its
# `optimum-cli` with --optimum-cli or VAANI_OPTIMUM_CLI.
HINDI_STT_REPO = "ARTPARK-IISc/whisper-tiny-vaani-hindi"
HINDI_STT_LICENSE = "Apache-2.0"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=",".join(STEPS),
                    help=f"comma-separated subset of {STEPS} (default: all)")
    ap.add_argument("--optimum-cli", default=os.environ.get("VAANI_OPTIMUM_CLI", "optimum-cli"),
                    help="optimum-cli of a torch venv (only for --only stt-hi)")
    args = ap.parse_args()
    only = set(args.only.split(","))
    ok = True

    # 1) Whisper tiny multilingual INT8 encoder + decoder + KV-cache decoder_with_past.
    if "stt" in only or "vocab" in only:
        ok = prepare_whisper(only) and ok
    if "stt-hi" in only:
        ok = prepare_whisper_hindi(args.optimum_cli) and ok
    if "vad" in only:
        ok = prepare_vad() and ok
    if "tts" in only:
        ok = prepare_tts() and ok
    if "mel" in only:
        prepare_mel()
    if "lexicon" in only:
        ok = prepare_lexicon() and ok
    ok = report_assets() and ok
    if not ok:
        log("FAILED: one or more assets missing/oversized")
        sys.exit(1)
    log("OK: all deployment assets prepared")


def prepare_whisper_hindi(optimum_cli):
    """Hindi-trained Whisper-tiny -> ONNX (optimum, separate decoders for the KV cache) ->
    fp16-stored encoder (fp32 compute) + INT8 decoders sharing one token table (ORT 1.18-safe).
    The encoder stays fp16-stored: INT8 there nearly doubled FLEURS CER (0.258 vs 0.132,
    evaluation_metrics.json -> hindi_asr_precision_ablation) for 4.8 MB saved."""
    log(f"Whisper Hindi specialist ({HINDI_STT_REPO}, {HINDI_STT_LICENSE})")
    out = "/tmp/raw_models/hindi_export"
    names = ("encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx")
    if not all(os.path.exists(os.path.join(out, n)) for n in names):
        cmd = [optimum_cli, "export", "onnx", "-m", HINDI_STT_REPO, "--task", "automatic-speech-recognition-with-past",
               "--no-post-process", "--opset", "14", out]
        log("  exporting with", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log("  ERROR: export failed:", (r.stdout + r.stderr)[-600:])
            return False
    fp16_storage(os.path.join(out, names[0]), f"{ASSETS}/vaani_stt_hi_encoder.fp16.onnx")
    quantize_decoder_shared_table(os.path.join(out, names[1]), f"{ASSETS}/vaani_stt_hi_decoder.int8.onnx")
    quantize_decoder_shared_table(os.path.join(out, names[2]), f"{ASSETS}/vaani_stt_hi_decoder_with_past.int8.onnx")
    for n in ("vaani_stt_hi_encoder.fp16.onnx", "vaani_stt_hi_decoder.int8.onnx", "vaani_stt_hi_decoder_with_past.int8.onnx"):
        set_metadata(f"{ASSETS}/{n}", HINDI_STT_REPO, HINDI_STT_LICENSE)
    return True


def prepare_whisper(only):
    ok = True
    log("Whisper tiny (multilingual, INT8)")
    # fp32 encoder, re-quantized MatMul-only below (the community INT8 encoder uses
    # ConvInteger with int8 weights, which ORT Android 1.18 cannot load).
    enc = hf_first([
        ("onnx-community/whisper-tiny", "onnx/encoder_model.onnx"),
        ("Xenova/whisper-tiny", "onnx/encoder_model.onnx"),
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
    # KV-cache decoder: one new token per call, consumes past_key_values.{0-3}.{decoder,encoder}.*
    dec_past = hf_first([
        ("onnx-community/whisper-tiny", "onnx/decoder_with_past_model_int8.onnx"),
        ("Xenova/whisper-tiny", "onnx/decoder_with_past_model_quantized.onnx"),
    ])
    if "stt" in only:
        if enc and dec and dec_past:
            quantize_matmul_int8(enc, f"{ASSETS}/vaani_stt_encoder.int8.onnx")
            set_metadata(f"{ASSETS}/vaani_stt_encoder.int8.onnx", "openai/whisper-tiny", "MIT")
            for src, name in ((dec, "vaani_stt_decoder.int8.onnx"),
                              (dec_past, "vaani_stt_decoder_with_past.int8.onnx")):
                shutil.copyfile(src, f"{ASSETS}/{name}")
                fuse_tied_logits_int8(f"{ASSETS}/{name}")
                set_metadata(f"{ASSETS}/{name}", "openai/whisper-tiny", "MIT")
        else:
            ok = False
            log("  ERROR: Whisper encoder/decoder/decoder_with_past not obtained")
    if "vocab" in only:
        if tok_json:
            build_vocab_json(tok_json, voc_json, f"{ASSETS}/vaani_vocab.json")
        else:
            ok = False
            log("  ERROR: Whisper tokenizer.json not obtained")
    return ok


def prepare_vad():
    ok = True
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
    return ok


def prepare_tts():
    ok = True
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
        fp16_storage(vits, f"{ASSETS}/vaani_tts_vits.fp16.onnx")
        set_metadata(f"{ASSETS}/vaani_tts_vits.fp16.onnx", "rhasspy/piper (hi_IN)", "MIT")
        legacy = f"{ASSETS}/vaani_tts_vits.int8.onnx"
        if os.path.exists(legacy):
            os.remove(legacy)          # ConvInteger(int8): unloadable on ORT 1.18, 3.2x slower
            log("  removed legacy vaani_tts_vits.int8.onnx")
    else:
        ok = False
        log("  ERROR: Piper VITS not obtained")
    if vits_cfg:
        shutil.copyfile(vits_cfg, f"{ASSETS}/vaani_tts_config.json")
    else:
        ok = False
        log("  ERROR: Piper config not obtained")
    return ok


def prepare_mel():
    log("Mel filterbank")
    fb = mel_filterbank()
    assert fb.shape == (N_MELS, N_FFT_BINS), fb.shape
    fb.astype("<f4").tofile(f"{ASSETS}/vaani_mel_filters.bin")
    log(f"  vaani_mel_filters.bin: shape={fb.shape}, "
        f"{os.path.getsize(f'{ASSETS}/vaani_mel_filters.bin')} bytes, sum={fb.sum():.4f}")


def report_assets():
    ok = True
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
    return ok


if __name__ == "__main__":
    main()
