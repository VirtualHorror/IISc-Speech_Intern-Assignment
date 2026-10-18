#!/usr/bin/env python3
# =============================================================================
#  VaaniEdge — Decoding-language experiment (Session 3): <|hi|> vs <|en|>
#  IISc Computational Intelligence Lab
#
#  Whisper-tiny often TRANSLATES Hinglish audio into English in <|hi|> mode. Hypothesis
#  tested: forcing <|en|> makes it transliterate (Roman Hinglish) instead. Every Hinglish
#  corpus utterance is synthesised with the shipped VITS (seeded), encoded once, then
#  decoded with both language tokens; CER is scored in Roman script against the
#  reference text. Result (seed 1234, 6 utterances x 6 samples): <|hi|> 0.594 vs
#  <|en|> 0.719 — <|en|> is worse (more translation/hallucination), so the app keeps <|hi|>.
#
#  Usage: python speech_lab/decode_language_ab.py [--runs 6] [--seed 1234]
#  (Needs the closed-loop fixture: run verify_closed_loop.py once first.)
# =============================================================================
import argparse
import os
import statistics
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_closed_loop as v  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    ort.set_seed(args.seed)
    import json
    fx = json.load(open(v.FIXTURE, encoding="utf-8"))
    filters = np.fromfile(os.path.join(v.ASSETS, "vaani_mel_filters.bin"), dtype="<f4").reshape(v.N_MELS, -1)
    dsp = v.HostDsp(v.build_host_dsp())
    vocab = v.Vocab(os.path.join(v.ASSETS, "vaani_vocab.json"))
    stt = v.WhisperOnnx(os.path.join(v.ASSETS, "vaani_stt_encoder.int8.onnx"),
                        os.path.join(v.ASSETS, "vaani_stt_decoder.int8.onnx"),
                        os.path.join(v.ASSETS, "vaani_stt_decoder_with_past.int8.onnx"), vocab, 4)
    tts = v.make_session(os.path.join(v.ASSETS, "vaani_tts_vits.fp16.onnx"), 4)
    cer = {"hi": [], "en": []}
    for utt in fx["utterances"]:
        if utt["lang"] != "hi" or utt.get("kind") == "heldout":
            continue
        for _ in range(args.runs):
            a22, *_ = v.synthesize(tts, utt)
            mel = dsp.log_mel(dsp.resample(a22, v.TTS_SR, v.STT_SR), filters)[0]
            hidden, _ = stt.encode(mel)
            for lang in ("hi", "en"):
                d = stt.decode_kv(hidden, lang, len(a22) / v.TTS_SR)
                cer[lang].append(v.score_display(vocab.decode(d["tokens"]), utt)["cer"])
    d = np.array(cer["en"]) - np.array(cer["hi"])
    print(f"<|hi|> Roman CER {statistics.fmean(cer['hi']):.4f} | <|en|> {statistics.fmean(cer['en']):.4f} | "
          f"paired en-hi {d.mean():+.4f} (en better {int((d < 0).sum())}, worse {int((d > 0).sum())}, n={d.size})")


if __name__ == "__main__":
    main()
