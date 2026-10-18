#!/usr/bin/env python3
# =============================================================================
#  VaaniEdge — quota-safe LLM smoke test: EXACTLY ONE streamed request per model
#  IISc Computational Intelligence Lab
#
#  Free-tier keys are precious (demo + evaluators), so this script never loops
#  or retries: every model in MODELS gets one request, in order, and the result
#  (TTFT, [ACTION:...] compliance, HTTP status, quota details) is recorded as-is.
#  Each provider's connection is first warmed with GET /models (metadata, no
#  generation quota), exactly like the app's pre-warm on VAD speech-start, so
#  TTFT is measured on a warm HTTPS connection.
#
#  The request is the app's real system prompt (read from PromptBuilder.kt) and
#  the alarm prompt as the app sends it: corrected transcript + raw ASR line.
#  Results -> speech_lab/evaluation_metrics.json ("llm_smoke_session3").
#  Keys come from local.properties / the environment and are never printed.
#
#  Usage: python speech_lab/llm_smoke_once.py [--only MODEL_ID ...]
#  (For repeated TTFT statistics use benchmark_llm.py — it costs 12+ calls/model.)
# =============================================================================
import argparse
import datetime as dt
import http.client
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PROMPT_KT = os.path.join(REPO, "app", "src", "main", "java", "in", "iisc", "vaani", "llm", "PromptBuilder.kt")
METRICS = os.path.join(HERE, "evaluation_metrics.json")

HOSTS = {"groq": "api.groq.com", "gemini": "generativelanguage.googleapis.com"}

# (provider, model id, extra request fields). One request each — never more.
MODELS = [
    ("groq", "llama-3.3-70b-versatile", {}),
    ("groq", "llama-3.1-8b-instant", {}),
    ("groq", "openai/gpt-oss-120b", {"reasoning_effort": "low", "include_reasoning": False}),
    ("groq", "openai/gpt-oss-20b", {"reasoning_effort": "low", "include_reasoning": False}),
    ("groq", "qwen/qwen3.8-27b", {"reasoning_effort": "none"}),
    ("gemini", "gemini-3.5-flash-lite", {"thinkingConfig": {"thinkingLevel": "minimal"}}),
    ("gemini", "gemini-3.5-flash", {"thinkingConfig": {"thinkingBudget": 0}}),
]

CONTEXT = """- Now: Sunday, 18 Oct 2026, 6:05 PM IST
- Bengaluru weather tomorrow (Mon 19 Oct): thunderstorms, 19.7–28.4°C, 84% chance of rain [Open-Meteo live]
- Today's schedule: 11:00 AM INT8 Speech Quantization Sync (done); 4:30 PM IISc Computational Intelligence Lab Review (done); 6:30 PM Edge DSP Telemetry Benchmark (NEXT, in 25 min)
- Next meeting: 6:30 PM Edge DSP Telemetry Benchmark
- Active alarms: none
- Active reminders: none"""

USER = ("[User spoke Hindi/Hinglish - reply in Roman Hinglish]\n"
        "Mujhe kal subah 7 baje ka alarm set karna hai\n"
        "[ASR raw output: Mohjee kaal subah, 7 bajyakal alarm sate karna hai]\n"
        "[Transcribed by on-device speech recognition; it may contain recognition errors - infer the intended request.]")
TAG = re.compile(r"^\s*\[ACTION:SET_ALARM\|07:00\|[^\]\n]+\]\s*$")


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


def request_parts(provider, model, extra, sysp, api_key):
    if provider == "groq":
        body = {"model": model, "stream": True, "temperature": 0.2,
                "max_completion_tokens": 512 if extra else 160,   # reasoning models share the budget
                "messages": [{"role": "system", "content": sysp}, {"role": "user", "content": USER}]}
        body.update(extra)
        return "/openai/v1/chat/completions", body, {"Authorization": f"Bearer {api_key}"}
    gen = {"temperature": 0.2, "maxOutputTokens": 160}
    gen.update(extra)
    body = {"systemInstruction": {"parts": [{"text": sysp}]},
            "contents": [{"role": "user", "parts": [{"text": USER}]}], "generationConfig": gen}
    return f"/v1beta/models/{model}:streamGenerateContent?alt=sse", body, {"x-goog-api-key": api_key}


def warm(conn, provider, api_key):
    path = "/openai/v1/models" if provider == "groq" else "/v1beta/models?pageSize=1"
    hdr = {"Authorization": f"Bearer {api_key}"} if provider == "groq" else {"x-goog-api-key": api_key}
    t0 = time.perf_counter()
    conn.request("GET", path, headers={**hdr, "User-Agent": "VaaniEdge-lab/3.0"})
    r = conn.getresponse()
    r.read()
    return r.status, round((time.perf_counter() - t0) * 1e3)


def quota_details(err):
    out = {}
    for d in (err.get("details") or []):
        t = d.get("@type", "")
        if t.endswith("QuotaFailure"):
            out["violations"] = [{k: v.get(k) for k in ("quotaMetric", "quotaId", "quotaValue")}
                                 for v in d.get("violations", [])]
        elif t.endswith("RetryInfo"):
            out["retry_delay"] = d.get("retryDelay")
    return out


def one_request(conn, provider, model, extra, sysp, api_key):
    path, body, hdr = request_parts(provider, model, extra, sysp, api_key)
    headers = {**hdr, "Content-Type": "application/json", "Accept": "text/event-stream",
               "User-Agent": "VaaniEdge-lab/3.0"}
    t0 = time.perf_counter()
    conn.request("POST", path, body=json.dumps(body).encode(), headers=headers)
    resp = conn.getresponse()
    headers_ms = round((time.perf_counter() - t0) * 1e3)
    res = {"provider": provider, "model": model, "params": extra, "http": resp.status, "headers_ms": headers_ms}
    rl = {k: v for k, v in resp.getheaders() if k.lower().startswith("x-ratelimit")}
    if rl:
        res["rate_limit_headers"] = rl
    if resp.status != 200:
        raw = resp.read().decode("utf-8", "replace")
        try:
            err = json.loads(raw).get("error", {})
        except ValueError:
            err = {"message": raw[:300]}
        res["error"] = {"message": (err.get("message") or "")[:400], "code": err.get("code"),
                        "status": err.get("status"), "type": err.get("type"), **quota_details(err)}
        return res
    ttft, parts, usage = None, [], None
    while True:
        line = resp.readline()
        if not line:
            break                                   # read to EOF so the connection is reusable
        line = line.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        d = json.loads(data)
        if provider == "groq":
            text = ((d.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
            usage = (d.get("x_groq") or {}).get("usage") or usage
        else:
            cand = (d.get("candidates") or [{}])[0]
            text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
            usage = d.get("usageMetadata") or usage
        if text:
            ttft = ttft if ttft is not None else round((time.perf_counter() - t0) * 1e3)
            parts.append(text)
    reply = "".join(parts)
    first_line = reply.strip().split("\n", 1)[0] if reply.strip() else ""
    res.update({
        "ttft_ms": ttft, "total_ms": round((time.perf_counter() - t0) * 1e3), "reply": reply,
        "action_tag_ok": bool(TAG.match(first_line)), "devanagari_in_reply": bool(re.search("[ऀ-ॿ]", reply)),
        "usage": usage,
    })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="restrict to these model ids (still one request each)")
    args = ap.parse_args()
    models = [m for m in MODELS if not args.only or m[1] in args.only]
    assert len({m[1] for m in models}) == len(models), "each model at most once"
    sysp = system_prompt()
    keys = {"groq": key("GROQ_API_KEY"), "gemini": key("GEMINI_API_KEY")}
    results, warmups = [], {}
    for provider in ("groq", "gemini"):
        todo = [m for m in models if m[0] == provider]
        if not todo:
            continue
        if not keys[provider]:
            results += [{"provider": provider, "model": m[1], "skipped": f"no {provider} key"} for m in todo]
            continue
        conn = http.client.HTTPSConnection(HOSTS[provider], timeout=30)
        try:
            warmups[provider] = dict(zip(("http", "cold_ms"), warm(conn, provider, keys[provider])))
            for _, model, extra in todo:
                try:
                    r = one_request(conn, provider, model, extra, sysp, keys[provider])
                except Exception as e:  # noqa: BLE001 — record, never retry
                    r = {"provider": provider, "model": model, "exception": f"{type(e).__name__}: {e}"}
                    conn.close()
                    conn = http.client.HTTPSConnection(HOSTS[provider], timeout=30)
                results.append(r)
                print(json.dumps({k: r.get(k) for k in ("model", "http", "ttft_ms", "total_ms", "action_tag_ok",
                                                        "error", "exception")}, ensure_ascii=False))
                if r.get("reply"):
                    print("   reply:", r["reply"].replace("\n", " | "))
                if r.get("rate_limit_headers"):
                    print("   limits:", r["rate_limit_headers"])
        finally:
            conn.close()

    report = json.load(open(METRICS, encoding="utf-8")) if os.path.exists(METRICS) else {}
    report["llm_smoke_session3"] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "method": ("one streamed request per model (no loops, no retries) from the build host (AWS, x86_64), "
                   "app system prompt + Hinglish alarm prompt with raw-ASR line; connection pre-warmed with "
                   "GET /models; ttft_ms = request sent -> first content token; n=1 per model"),
        "warmup": warmups,
        "results": results,
    }
    tmp = METRICS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, METRICS)
    print(f"[smoke] wrote {METRICS} (llm_smoke_session3)")


if __name__ == "__main__":
    main()
