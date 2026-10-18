#!/usr/bin/env python3
# =============================================================================
#  VaaniEdge — Streaming LLM benchmark (TTFT + tool-tag compliance)
#  IISc Computational Intelligence Lab
#
#  Sends the app's REAL system prompt (the template is read straight out of
#  app/src/main/java/in/iisc/vaani/llm/PromptBuilder.kt) with a grounded context
#  block to each streaming model, using the transcripts the on-device STT actually
#  produced in the closed loop (noisy Hinglish included), and measures
#  time-to-first-token, total stream time, and whether alarm/reminder turns emit
#  the [ACTION:...] tag the app parses. Results -> evaluation_metrics.json
#  ("llm_benchmark"). Keys are read from local.properties / the environment and
#  are never written anywhere.
#
#  Usage: python speech_lab/benchmark_llm.py [--runs 3]
#  (Measured from the build host, not the phone: network RTT differs on-device.)
# =============================================================================
import argparse
import datetime as dt
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PROMPT_KT = os.path.join(REPO, "app", "src", "main", "java", "in", "iisc", "vaani", "llm", "PromptBuilder.kt")
METRICS = os.path.join(HERE, "evaluation_metrics.json")

CONTEXT = """- Now: Sunday, 18 Oct 2026, 3:42 PM IST
- Bengaluru weather tomorrow (Mon 19 Oct): thunderstorms, 19.7–28.4°C, 84% chance of rain [Open-Meteo live]
- Today's schedule: 11:00 AM INT8 Speech Quantization Sync (done); 4:30 PM IISc Computational Intelligence Lab Review (NEXT, in 48 min); 6:00 PM Edge DSP Telemetry Benchmark
- Next meeting: 4:30 PM IISc Computational Intelligence Lab Review
- Active alarms: none
- Active reminders: none"""

# (prompt id, LID language, transcript as recognised on-device in the closed loop after the
#  app's HinglishCodec.cleanAsrText ("7-7 baji" -> "7 baji"), expected tag)
TURNS = [
    ("P1", "en", "What is the weather forecast for Bangalore tomorrow?", None),
    ("P2", "en", "Can you set a reminder for my meeting at 5 p.m.", "SET_REMINDER|17:00"),
    ("P3", "hi", "muji kalsuba, 7 baji ka alarm is 8 karna hai", "SET_ALARM|07:00"),
    ("P4", "hi", "Today's schedule is my next meeting.", None),
]

MODELS = [
    ("gemini", "gemini-3.5-flash-lite", {"thinkingLevel": "minimal"}),
    ("gemini", "gemini-3.5-flash", {"thinkingBudget": 0}),
    ("groq", "llama-3.3-70b-versatile", None),
    ("groq", "llama-3.1-8b-instant", None),
]


def key(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        with open(os.path.join(REPO, "local.properties"), encoding="utf-8") as f:
            for line in f:
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def system_prompt():
    src = open(PROMPT_KT, encoding="utf-8").read()
    tmpl = re.search(r'systemPrompt\(tools: ToolSnapshot\): String = """(.*?)""".trim\(\)', src, re.S).group(1)
    return tmpl.replace("${ToolContext.render(tools)}", CONTEXT).strip()


def user_message(lang, text):
    head = "[User spoke Hindi/Hinglish - reply in Roman Hinglish]\n" if lang == "hi" else "[User spoke English - reply in English]\n"
    return head + text + "\n[Transcribed by on-device speech recognition; it may contain recognition errors - infer the intended request.]"


def stream(provider, model, think, sysp, user, api_key):
    if provider == "gemini":
        gen = {"temperature": 0.2, "maxOutputTokens": 160}
        if think:
            gen["thinkingConfig"] = think
        body = {"systemInstruction": {"parts": [{"text": sysp}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}], "generationConfig": gen}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    else:
        body = {"model": model, "stream": True, "temperature": 0.2, "max_tokens": 160,
                "messages": [{"role": "system", "content": sysp}, {"role": "user", "content": user}]}
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    t0 = time.perf_counter()
    ttft, out = None, []
    with urllib.request.urlopen(req, timeout=40) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if provider == "gemini":
                parts = d.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            else:
                text = (d.get("choices", [{}])[0].get("delta") or {}).get("content") or ""
            if text:
                ttft = ttft or (time.perf_counter() - t0) * 1e3
                out.append(text)
    return ttft, (time.perf_counter() - t0) * 1e3, "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--spacing", type=float, default=4.0, help="seconds between calls (rate limits)")
    args = ap.parse_args()
    sysp = system_prompt()
    keys = {"gemini": key("GEMINI_API_KEY"), "groq": key("GROQ_API_KEY")}
    results = {}
    for provider, model, think in MODELS:
        if not keys[provider]:
            results[model] = {"skipped": f"no {provider} key"}
            print(f"[llm] {model}: skipped (no {provider} key)")
            continue
        ttfts, totals, errors, tags_ok, tags_need, replies = [], [], [], 0, 0, {}
        for _ in range(args.runs):
            for pid, lang, text, tag in TURNS:
                time.sleep(args.spacing)             # stay under free-tier per-minute quotas
                try:
                    ttft, total, reply = stream(provider, model, think, sysp, user_message(lang, text), keys[provider])
                except urllib.error.HTTPError as e:
                    errors.append(f"HTTP {e.code}")
                    continue
                except Exception as e:  # noqa: BLE001
                    errors.append(type(e).__name__)
                    continue
                if ttft:
                    ttfts.append(ttft)
                totals.append(total)
                if tag:
                    tags_need += 1
                    tags_ok += f"[ACTION:{tag}" in reply.replace(" ", "")
                replies.setdefault(pid, reply)
        results[model] = {
            "provider": provider, "thinking": think,
            "ttft_ms": {"median": round(statistics.median(ttfts)), "min": round(min(ttfts)), "max": round(max(ttfts))} if ttfts else None,
            "total_ms_median": round(statistics.median(totals)) if totals else None,
            "calls_ok": len(totals), "calls_failed": len(errors), "errors": sorted(set(errors)),
            "tool_tag_compliance": f"{tags_ok}/{tags_need}",
            "sample_replies": replies,
        }
        print(f"[llm] {model}: TTFT median {results[model]['ttft_ms']} tags {tags_ok}/{tags_need} errors {errors}")

    report = json.load(open(METRICS, encoding="utf-8")) if os.path.exists(METRICS) else {}
    report["llm_benchmark"] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": ("Measured from the build host with the app's real system prompt and the noisy transcripts the "
                 "on-device STT produced in the closed loop. gemini-2.5-flash (requested in the brief) answers "
                 "HTTP 404 'no longer available to new users', so the app uses the 3.5 generation."),
        "runs_per_prompt": args.runs,
        "models": results,
    }
    tmp = METRICS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp, METRICS)
    print(f"[llm] wrote {METRICS}")


if __name__ == "__main__":
    main()
