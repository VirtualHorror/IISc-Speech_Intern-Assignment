#!/usr/bin/env python3
# =============================================================================
#  VaaniEdge — Hindi / Hinglish ASR model evaluation (real + synthetic speech)
#  IISc Computational Intelligence Lab
#
#  Compares Whisper-architecture checkpoints (ONNX) exactly as the app decodes:
#  C++ log-Mel -> encoder -> KV-cached greedy decoder forced to <|hi|> with the
#  app's masks (verify_closed_loop.WhisperOnnx). Two test sets:
#    * FLEURS hi_in test (real read speech, Devanagari references; CC-BY-4.0),
#      a deterministic subset (first N ids);
#    * the closed-loop Hindi/Hinglish commands (corpus + held-out), synthesised
#      once with the shipped VITS voice (seeded) and fed to every model.
#  Scores:
#    * native: CER/WER in Devanagari (punctuation, nukta, chandrabindu, digits
#      normalised) — the standard metric for Devanagari-output models;
#    * display: CER/WER of the Roman text the app shows, i.e. hypothesis AND
#      reference both through the app's Kotlin HinglishCodec.displayTranscript
#      (SttDisplayExportTest bridge) — comparable across Devanagari- and
#      Roman-output models;
#    * latency: encoder ms and decoder ms/token on this host.
#  Results -> speech_lab/evaluation_metrics.json ("hindi_asr_eval").
#
#  Usage:
#    python speech_lab/eval_hindi_asr.py --fleurs-dir /tmp/vaani_eval/fleurs_hi \
#        --models shipped=ENC,DEC,PAST tiny_hi=/dir/with/encoder_model.onnx... [--fleurs-n 100]
# =============================================================================
import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
import unicodedata


import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_closed_loop as v  # noqa: E402

NAMES = ("encoder_model", "decoder_model", "decoder_with_past_model")
RAW = os.path.join(v.BUILD, "hindi_asr_raw{}.json")        # every decoded transcript, for --rescore


def model_paths(spec):
    """'DIR' (optimum export layout) or 'ENC,DEC,PAST'."""
    if "," in spec:
        enc, dec, past = spec.split(",")
        return enc, dec, past
    def pick(stem):
        for suffix in (".int8.onnx", "_int8.onnx", ".onnx"):
            p = os.path.join(spec, stem + suffix)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"{stem} in {spec}")
    return tuple(pick(n) for n in NAMES)


def raw_path(args):
    return RAW.format("" if args.key == "hindi_asr_eval" else "_" + args.key)


def read_wav16(path):
    """RIFF/WAVE reader for 16-bit PCM and 32-bit IEEE float (FLEURS ships the latter)."""
    data = open(path, "rb").read()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", path
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = (int.from_bytes(body[0:2], "little"), int.from_bytes(body[2:4], "little"),
                   int.from_bytes(body[4:8], "little"), int.from_bytes(body[14:16], "little"))
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    tag, ch, sr, bits = fmt
    if tag == 0xFFFE:                                    # WAVE_FORMAT_EXTENSIBLE: assume the sub-format
        tag = 3 if bits == 32 else 1
    if tag == 3 and bits == 32:
        x = np.frombuffer(pcm, dtype="<f4").astype(np.float32)
    elif tag == 1 and bits == 16:
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise ValueError(f"{path}: format {tag}, {bits}-bit")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def norm_dev(s):
    """Devanagari normalisation for native CER/WER (keeps letters, marks, numbers)."""
    s = unicodedata.normalize("NFC", s).replace("़", "").replace("ँ", "ं")
    s = "".join(chr(ord(c) - 0x0966 + 0x30) if "०" <= c <= "९" else c for c in s)
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return " ".join(s.split())


def rates(ref, hyp):
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    cer = v.edit_distance(list(r), list(h)) / max(1, len(r))
    wer = v.edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split()))
    return cer, wer


def fleurs_subset(fleurs_dir, n):
    rows = list(csv.reader(open(os.path.join(fleurs_dir, "data", "hi_in", "test.tsv"), encoding="utf-8"),
                           delimiter="\t", quoting=csv.QUOTE_NONE))
    wav_dir = os.path.join(fleurs_dir, "test_wav", "test")
    items, seen = [], set()
    for r in sorted(rows, key=lambda r: (int(r[0]), r[1])):
        if r[0] in seen:                       # FLEURS repeats a sentence for several speakers
            continue
        path = os.path.join(wav_dir, r[1])
        if os.path.exists(path):
            seen.add(r[0])
            items.append({"id": f"fleurs-{r[0]}", "wav": path, "ref": r[2]})
        if len(items) >= n:
            break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", help="name=DIR or name=ENC,DEC,PAST")
    ap.add_argument("--rescore", action="store_true",
                    help="re-run only the Kotlin display pass + metrics on the transcripts of the last run")
    ap.add_argument("--fleurs-dir", default="/tmp/vaani_eval/fleurs_hi")
    ap.add_argument("--fleurs-n", type=int, default=100)
    ap.add_argument("--runs", type=int, default=3, help="seeded VITS samples per closed-loop utterance")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=v.METRICS)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--key", default="hindi_asr_eval", help="evaluation_metrics.json block (and raw-file suffix)")
    args = ap.parse_args()
    if args.rescore:
        saved = json.load(open(raw_path(args), encoding="utf-8"))
        return score_and_write(saved["results"], saved["samples"], saved["n_fleurs"], saved["runs"], args)
    ort.set_seed(args.seed)

    filters = np.fromfile(os.path.join(v.ASSETS, "vaani_mel_filters.bin"), dtype="<f4").reshape(v.N_MELS, -1)
    dsp = v.HostDsp(v.build_host_dsp())
    vocab = v.Vocab(os.path.join(v.ASSETS, "vaani_vocab.json"))

    # ---- inputs: real FLEURS speech + seeded synthetic commands (shared by every model) ----
    samples = []
    for it in fleurs_subset(args.fleurs_dir, args.fleurs_n):
        x, sr = read_wav16(it["wav"])
        if sr != v.STT_SR:
            x = dsp.resample(x, sr, v.STT_SR)
        x = x[: v.N_SAMPLES]
        samples.append({"set": "fleurs", "id": it["id"], "ref": it["ref"], "lang": "hi",
                        "mel": dsp.log_mel(x, filters)[0], "sec": len(x) / v.STT_SR})
    fixture = json.load(open(v.FIXTURE, encoding="utf-8"))
    tts = v.make_session(os.path.join(v.ASSETS, "vaani_tts_vits.fp16.onnx"), args.threads, mem_pattern=False)
    for utt in fixture["utterances"]:
        if utt["lang"] != "hi":
            continue
        for r in range(args.runs):
            a22, *_ = v.synthesize(tts, utt)
            a16 = dsp.resample(a22, v.TTS_SR, v.STT_SR)
            samples.append({"set": "heldout" if utt.get("kind") == "heldout" else "corpus", "id": f"{utt['id']}#{r}",
                            "ref": utt["text"], "utt": utt, "lang": "hi", "mel": dsp.log_mel(a16, filters)[0],
                            "sec": len(a22) / v.TTS_SR})
    n_fleurs = sum(s["set"] == "fleurs" for s in samples)
    v.log(f"{n_fleurs} FLEURS + {len(samples) - n_fleurs} synthetic samples; {len(args.models)} models")

    # ---- decode every sample with every model ----
    results = {}
    for spec in args.models:
        name, path_spec = spec.split("=", 1)
        enc, dec, past = model_paths(path_spec)
        stt = v.WhisperOnnx(enc, dec, past, vocab, args.threads)
        stt.encode(v.whisper_log_mel_reference(np.zeros(v.STT_SR, np.float32), filters))   # warm-up
        rows = []
        for s in samples:
            hidden, enc_ms = stt.encode(s["mel"])
            d = stt.decode_kv(hidden, "hi", s["sec"])
            rows.append({"id": s["id"], "set": s["set"], "text": vocab.decode(d["tokens"]), "enc_ms": enc_ms,
                         "ms_per_step": d["decode_ms"] / max(1, d["decoder_calls"]), "lid_step0": d["detected"]})
        size = sum(os.path.getsize(p) for p in (enc, dec, past)) / 1e6
        results[name] = {"files": [os.path.basename(p) for p in (enc, dec, past)], "size_mb": round(size, 1), "rows": rows}
        fl = [r for r in rows if r["set"] == "fleurs"]
        v.log(f"{name:16s} decoded {len(rows)} | enc {statistics.median(r['enc_ms'] for r in rows):.0f} ms | "
              f"{statistics.median(r['ms_per_step'] for r in rows):.1f} ms/step | e.g. {fl[0]['text'][:70] if fl else ''}")

    meta = [{"id": s["id"], "set": s["set"], "ref": s["ref"]} for s in samples]
    os.makedirs(v.BUILD, exist_ok=True)
    with open(raw_path(args), "w", encoding="utf-8") as f:
        json.dump({"results": results, "samples": meta, "n_fleurs": n_fleurs, "runs": args.runs}, f, ensure_ascii=False)
    score_and_write(results, meta, n_fleurs, args.runs, args)


def score_and_write(results, samples, n_fleurs, runs, args):
    # ---- the app's display pipeline (Kotlin) on hypotheses and FLEURS references ----
    items = [{"key": f"{n}|{r['id']}", "raw": r["text"], "lang": "hi"} for n, res in results.items() for r in res["rows"]]
    items += [{"key": f"REF|{s['id']}", "raw": s["ref"], "lang": "hi"} for s in samples if s["set"] == "fleurs"]
    disp = v.kotlin_display(items) or {}

    refs = {s["id"]: s for s in samples}
    first_fleurs = {s["id"] for s in samples if s["set"] == "fleurs"}
    first_fleurs = set(sorted(first_fleurs)[:3])
    summary = {}
    for name, res in results.items():
        per_set = {}
        for set_name in ("fleurs", "corpus", "heldout"):
            rows = [r for r in res["rows"] if r["set"] == set_name]
            if not rows:
                continue
            native, display = [], []
            for r in rows:
                s = refs[r["id"]]
                hyp_disp = (disp.get(f"{name}|{r['id']}") or {}).get("display", r["text"])
                if set_name == "fleurs":
                    native.append(rates(norm_dev(s["ref"]), norm_dev(r["text"])))
                    ref_disp = (disp.get(f"REF|{r['id']}") or {}).get("display", s["ref"])
                    display.append(rates(v.norm_lat(ref_disp), v.norm_lat(hyp_disp)))
                else:
                    sc = v.score_display(hyp_disp, {"text": s["ref"]})
                    display.append((sc["cer"], sc["wer"]))
            block = {"n": len(rows),
                     "display_cer": round(statistics.fmean(c for c, _ in display), 4),
                     "display_wer": round(statistics.fmean(w for _, w in display), 4)}
            if native:
                block["native_cer"] = round(statistics.fmean(c for c, _ in native), 4)
                block["native_wer"] = round(statistics.fmean(w for _, w in native), 4)
            if set_name == "fleurs":
                block["step0_lid_hi_pct"] = round(100.0 * statistics.fmean(r["lid_step0"] == "hi" for r in rows), 1)
            per_set[set_name] = block
        summary[name] = {
            "size_mb": res["size_mb"], "files": res["files"],
            "encoder_ms_median": round(statistics.median(r["enc_ms"] for r in res["rows"]), 1),
            "decoder_ms_per_step_median": round(statistics.median(r["ms_per_step"] for r in res["rows"]), 2),
            **per_set,
            "samples": {r["id"]: {"raw": r["text"], "display": (disp.get(f"{name}|{r['id']}") or {}).get("display")}
                        for r in res["rows"] if r["id"].endswith("#0") or r["id"] in first_fleurs},
        }
        log_sets = {k: {kk: vv for kk, vv in b.items() if kk != "n"} for k, b in per_set.items()}
        v.log(f"{name:16s} {json.dumps(log_sets)}")

    report_block = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "method": (f"app-identical decoding (C++ log-Mel, KV-cached greedy, forced <|hi|>, app masks); "
                   f"FLEURS hi_in test first {n_fleurs} sentence ids (real read speech, CC-BY-4.0) + closed-loop "
                   f"Hindi/Hinglish commands x {runs} seeded VITS samples; native = Devanagari CER/WER; "
                   f"display = Roman CER/WER after the app's Kotlin displayTranscript on hypothesis and reference; "
                   f"host {v.cpu_model()}, onnxruntime {ort.__version__}, {args.threads} threads"),
        "models": summary,
    }
    if not args.no_write:
        report = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {}
        report[args.key] = report_block
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=float)
        os.replace(tmp, args.out)
        v.log(f"wrote {args.out} ({args.key})")


if __name__ == "__main__":
    t0 = time.time()
    main()
    v.log(f"done in {time.time() - t0:.0f} s")
