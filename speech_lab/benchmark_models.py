#!/usr/bin/env python3
# =============================================================================
#  VaaniEdge — Model Benchmark & Tensor-Signature Verification
#  IISc Computational Intelligence Lab
#
#  Opens every deployment ONNX graph with onnxruntime.InferenceSession, records
#  exact input/output tensor names + shapes + dtypes (so the Kotlin OrtSession
#  code binds the EXACT names), computes parameter counts and INT8 compression,
#  and writes a projected Snapdragon 8+ Gen 1 RTF.
#
#  Output: speech_lab/evaluation_metrics.json  (+ stdout tensor signatures).
#
#  NOTE: the "projected_*" figures are analytical planning numbers only. MEASURED
#  host latencies/accuracy come from verify_closed_loop.py ("closed_loop"), live
#  on-device numbers from the app's telemetry HUD / logcat (tag VaaniTelemetry).
#  This script only rewrites the "models" and "totals" keys of the report.
# =============================================================================
import glob
import json
import os

import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "app", "src", "main", "assets", "models"))
RAW = "/tmp/raw_models"

# Snapdragon 8+ Gen 1 (Cortex-X2 @ ~3.2GHz + A710). Rough sustained INT8 CPU
# throughput used only for an order-of-magnitude RTF projection (XNNPACK, 4 big
# cores). These are planning numbers, not measurements.
SD8PLUS_INT8_GOPS = 55.0


def tensor_sig(t):
    shape = [d if isinstance(d, int) else (d or "dyn") for d in t.shape]
    return {"name": t.name, "shape": shape, "type": t.type}


def param_count(path):
    try:
        m = onnx.load(path, load_external_data=False)
        total = 0
        for init in m.graph.initializer:
            n = 1
            for d in init.dims:
                n *= d
            total += n
        return total
    except Exception:  # noqa: BLE001
        return 0


def raw_fp32_size(basename_contains):
    for p in glob.glob(f"{RAW}/**/*.onnx", recursive=True):
        b = os.path.basename(p).lower()
        if basename_contains in b and not any(t in b for t in ("int8", "quant", "fp16", "q4", "bnb4", "uint8", "merged")):
            return os.path.getsize(p)
    return None


def project_rtf(params, kind):
    # Analytical estimate only (NO on-device run this session). FLOPs ~= 2 * params
    # * sequence_steps, over a rough sustained INT8 throughput. Returns (gflops, rtf).
    ref_audio_s = 5.0
    if kind == "stt_encoder":
        gflops = 2 * params * 1500 / 1e9      # fixed 30 s Mel window (1500 frames)
        return round(gflops, 2), round(gflops / SD8PLUS_INT8_GOPS / ref_audio_s, 3)
    if kind == "stt_decoder_with_past":
        gflops = 2 * params * 32 / 1e9        # ~32 cached steps, one token each
        return round(gflops, 2), round(gflops / SD8PLUS_INT8_GOPS / ref_audio_s, 3)
    if kind == "stt_decoder":
        gflops = 2 * params * 32 / 1e9        # ~32 greedy tokens with KV-cache (Session 2)
        return round(gflops, 2), round(gflops / SD8PLUS_INT8_GOPS / ref_audio_s, 3)
    if kind == "tts":
        gflops = 2 * params * 160 / 1e9       # ~160 VITS flow steps per clause (~1.5 s audio)
        return round(gflops, 2), round(gflops / SD8PLUS_INT8_GOPS / 1.5, 3)
    gflops = 2 * params * 64 / 1e9            # VAD: per-frame, negligible
    return round(gflops, 3), round(gflops / SD8PLUS_INT8_GOPS / ref_audio_s, 4)


def analyze(path, kind):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ins = [tensor_sig(t) for t in sess.get_inputs()]
    outs = [tensor_sig(t) for t in sess.get_outputs()]
    size = os.path.getsize(path)
    params = param_count(path)
    fp32 = raw_fp32_size(
        {"stt_encoder": "encoder_model.onnx", "stt_decoder": "decoder_model.onnx",
         "stt_decoder_with_past": "decoder_with_past_model.onnx", "tts": "medium", "vad": "vad"}.get(kind, ""))
    theo_fp32 = params * 4
    gflops, rtf = project_rtf(params, kind)
    entry = {
        "file": os.path.basename(path),
        "size_mb": round(size / 1e6, 3),
        "param_count": params,
        "theoretical_fp32_mb": round(theo_fp32 / 1e6, 3) if params > 1000 else None,
        "int8_size_reduction_pct": (
            round(100.0 * (1 - size / theo_fp32), 2) if (params > 1000 and size < theo_fp32) else None),
        "inputs": ins,
        "outputs": outs,
        "projected_gflops_per_inference": gflops,
        "projected_rtf_sd8plus_gen1": rtf,
    }
    if fp32:
        entry["measured_fp32_mb"] = round(fp32 / 1e6, 3)
        entry["measured_reduction_pct"] = round(100.0 * (1 - size / fp32), 2)

    print(f"\n=== {os.path.basename(path)} ({kind}) ===")
    print(f"  size={entry['size_mb']}MB params={params:,} "
          f"proj_RTF={entry['projected_rtf_sd8plus_gen1']}")
    for t in ins:
        print(f"  IN   {t['name']:28s} {t['type']:16s} {t['shape']}")
    for t in outs:
        print(f"  OUT  {t['name']:28s} {t['type']:16s} {t['shape']}")
    return entry


def main():
    targets = [
        ("vaani_stt_encoder.int8.onnx", "stt_encoder"),
        ("vaani_stt_decoder.int8.onnx", "stt_decoder"),
        ("vaani_stt_decoder_with_past.int8.onnx", "stt_decoder_with_past"),
        ("vaani_vad.onnx", "vad"),
        ("vaani_tts_vits.fp16.onnx", "tts"),
    ]
    out = os.path.join(HERE, "evaluation_metrics.json")
    report = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {}
    report["target_device"] = "OnePlus 11R 5G (CPH2487) — Snapdragon 8+ Gen 1, arm64-v8a"
    report["rtf_disclaimer"] = ("models.*.projected_* are analytical planning numbers; measured host numbers "
                                "are in closed_loop, live phone numbers in the app HUD")
    report["models"] = {}
    report["totals"] = {}
    total_size = 0
    for name, kind in targets:
        path = os.path.join(ASSETS, name)
        if not os.path.exists(path):
            print(f"  MISSING: {name}")
            continue
        entry = analyze(path, kind)
        report["models"][name] = entry
        total_size += entry["size_mb"]

    # Non-model assets.
    for extra in ("vaani_mel_filters.bin", "vaani_vocab.json", "vaani_tts_config.json"):
        p = os.path.join(ASSETS, extra)
        if os.path.exists(p):
            report["models"].setdefault("_assets", {})[extra] = round(os.path.getsize(p) / 1e6, 4)

    report["totals"]["model_size_mb"] = round(total_size, 2)
    report["totals"]["assets_dir_mb"] = round(
        sum(os.path.getsize(os.path.join(ASSETS, f)) for f in os.listdir(ASSETS)) / 1e6, 2)

    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out)
    print(f"\nWrote {out}")
    print(f"Total model size: {report['totals']['model_size_mb']} MB; "
          f"assets dir: {report['totals']['assets_dir_mb']} MB")


if __name__ == "__main__":
    main()
