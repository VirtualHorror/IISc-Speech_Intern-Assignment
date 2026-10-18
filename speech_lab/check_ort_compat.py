#!/usr/bin/env python3
# =============================================================================
#  VaaniEdge — Runtime compatibility gate for the shipped ONNX graphs
#  IISc Computational Intelligence Lab
#
#  Loads and runs every app/src/main/assets/models/*.onnx with whatever
#  onnxruntime is installed. Run it with the SAME version the APK ships
#  (com.microsoft.onnxruntime:onnxruntime-android:1.18.0):
#
#     python3 -m venv --without-pip /tmp/vaani_venv_ort118
#     curl -sSL https://bootstrap.pypa.io/get-pip.py | /tmp/vaani_venv_ort118/bin/python -
#     /tmp/vaani_venv_ort118/bin/pip install onnxruntime==1.18.0 "numpy<2"
#     /tmp/vaani_venv_ort118/bin/python speech_lab/check_ort_compat.py [--write]
#
#  This is the host-side stand-in for an on-device load test: it caught that the
#  Session-1 INT8 encoder/VITS (ConvInteger with int8 weights) cannot be loaded by
#  ORT <= 1.22, i.e. the app would have failed at model-load time on the phone.
# =============================================================================
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "app", "src", "main", "assets", "models"))


def dummy_feeds(sess):
    feeds = {}
    for i in sess.get_inputs():
        if i.name == "input_features":
            feeds[i.name] = np.zeros((1, 80, 3000), np.float32)
        elif i.name == "encoder_hidden_states":
            feeds[i.name] = np.zeros((1, 1500, 384), np.float32)
        elif i.name == "input_ids":
            feeds[i.name] = np.array([[50258]], np.int64)
        elif i.name.startswith("past_key_values"):
            n = 1500 if ".encoder." in i.name else 4
            feeds[i.name] = np.zeros((1, 6, n, 64), np.float32)
        elif i.name == "input" and "int64" in i.type:                  # VITS phoneme ids
            feeds[i.name] = np.array([[1, 0, 25, 0, 14, 0, 2]], np.int64)
        elif i.name == "input_lengths":
            feeds[i.name] = np.array([7], np.int64)
        elif i.name == "scales":
            feeds[i.name] = np.array([0.667, 1.0, 0.8], np.float32)
        elif i.name == "input":                                        # Silero VAD audio
            feeds[i.name] = np.zeros((1, 512), np.float32)
        elif i.name == "state":
            feeds[i.name] = np.zeros((2, 1, 128), np.float32)
        elif i.name == "sr":
            feeds[i.name] = np.array(16000, np.int64)
    return feeds


def main():
    report = {"onnxruntime": ort.__version__, "models": {}}
    ok = True
    for name in sorted(f for f in os.listdir(ASSETS) if f.endswith(".onnx")):
        path = os.path.join(ASSETS, name)
        try:
            t0 = time.perf_counter()
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            load_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            outs = sess.run(None, dummy_feeds(sess))
            run_ms = (time.perf_counter() - t0) * 1e3
            report["models"][name] = {"ok": True, "load_ms": round(load_ms), "run_ms": round(run_ms, 1),
                                      "output0_shape": list(outs[0].shape)}
            print(f"  OK    {name:42s} load {load_ms:6.0f} ms  run {run_ms:7.1f} ms  out0 {list(outs[0].shape)}")
        except Exception as e:  # noqa: BLE001
            ok = False
            report["models"][name] = {"ok": False, "error": str(e)[:300]}
            print(f"  FAIL  {name:42s} {str(e)[:160]}")
    report["all_ok"] = ok
    print(json.dumps({"onnxruntime": report["onnxruntime"], "all_ok": ok}))
    if "--write" in sys.argv:                      # merge into speech_lab/evaluation_metrics.json
        metrics = os.path.join(HERE, "evaluation_metrics.json")
        data = json.load(open(metrics, encoding="utf-8")) if os.path.exists(metrics) else {}
        data.setdefault("ort_runtime_compat", {})[ort.__version__] = report
        tmp = metrics + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, metrics)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
