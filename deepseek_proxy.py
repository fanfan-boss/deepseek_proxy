# ╔══════════════════════════════════════════════════════════════╗
# ║           Android Studio Agent 配置方法                       ║
# ╠══════════════════════════════════════════════════════════════╣
# ║                                                            ║
# ║  1. 打开 File → Settings → Tools → AI → Model Providers    ║
# ║                                                            ║
# ║  2. 添加一个 Provider（URL Schema 选 OpenAI-compatible）：  ║
# ║                                                            ║
# ║     ┌─────────────────────────────────────────────┐         ║
# ║     │  Description   │  deepseek-proxy            │         ║
# ║     │  URL           │  http://127.0.0.2:8081  │         ║
# ║     │  URL Schema    │  OpenAI-compatible  ✅     │         ║
# ║     │  API Key       │  你的 DeepSeek API Key     │         ║
# ║     │  Model         │  deepseek-v4-flash         │         ║
# ║     └─────────────────────────────────────────────┘         ║
# ║                                                            ║
# ║  3. 点 Apply / OK，即可在 Agent 中使用                      ║
# ║                                                            ║
# ║  模型选择:  deepseek-v4-flash → 快速对话（推荐）           ║
# ║            deepseek-v4-pro   → 深度推理                    ║
# ║                                                            ║
# ╚══════════════════════════════════════════════════════════════╝

from flask import Flask, request, Response, stream_with_context
import requests
import json
import logging
import os
import threading

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek_proxy.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)

log = logging.getLogger("proxy")
log.info("--- Proxy started ---")

app = Flask(__name__)
DEEPSEEK_BASE = "https://api.deepseek.com"

upstream_kv_hit = 0
upstream_kv_miss = 0

reasoning_cache = {}
reasoning_cache_hits = 0
reasoning_cache_misses = 0


def _msg_signature(msg):
    tcs = msg.get("tool_calls")
    if tcs and isinstance(tcs, list):
        ids = tuple(sorted(tc.get("id", "") for tc in tcs if isinstance(tc, dict) and tc.get("id")))
        if ids:
            content = msg.get("content", "")
            return ("tool", content, ids)
    return None


def _cache_reasoning(msg):
    sig = _msg_signature(msg)
    rc = msg.get("reasoning_content")
    if sig and rc:
        reasoning_cache[sig] = rc
        return True
    return False


def _inject_reasoning_content(messages):
    global reasoning_cache_hits, reasoning_cache_misses
    for msg in messages:
        sig = _msg_signature(msg)
        if sig and "reasoning_content" not in msg:
            cached = reasoning_cache.get(sig)
            if cached:
                msg["reasoning_content"] = cached
                reasoning_cache_hits += 1
                log.debug("Injected cached reasoning_content (hit) for tool_calls assistant message")
            else:
                reasoning_cache_misses += 1
                log.debug("Cache miss for tool_calls assistant message")


def _enable_thinking(body):
    if "thinking" not in body:
        body["thinking"] = {"type": "enabled"}
    if "reasoning_effort" not in body:
        body["reasoning_effort"] = "high"


def _disable_thinking(body):
    body["thinking"] = {"type": "disabled"}
    body.pop("reasoning_effort", None)


# ──────────────────────────────
# Gemini ↔ OpenAI 格式转换
# ──────────────────────────────

GEMINI_MODEL_MAP = {
    "gemini-2.0-flash": "deepseek-v4-flash",
    "gemini-2.0-pro": "deepseek-v4-pro",
    "gemini-1.5-pro": "deepseek-v4-pro",
    "gemini-1.5-flash": "deepseek-v4-flash",
    "gemini-pro": "deepseek-v4-pro",
}

# Gemini roles -> OpenAI roles mapping
_GEMINI_ROLE_MAP = {"user": "user", "model": "assistant", "system": "system", "function": "function"}
_OPENAI_ROLE_MAP = {v: k for k, v in _GEMINI_ROLE_MAP.items()}


def _gemini_to_openai(gemini_body):
    """Convert Gemini request body to OpenAI-compatible format."""
    openai_messages = []

    # system instruction
    sys_inst = gemini_body.get("systemInstruction")
    if sys_inst:
        parts = sys_inst.get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        if text:
            openai_messages.append({"role": "system", "content": text})

    # contents
    for c in gemini_body.get("contents", []):
        role = _GEMINI_ROLE_MAP.get(c.get("role", "user"), "user")
        parts = c.get("parts", [])
        text = ""
        for p in parts:
            if "text" in p:
                text += p["text"]
            elif "functionCall" in p:
                text += json.dumps(p["functionCall"], ensure_ascii=False)
            elif "functionResponse" in p:
                text += json.dumps(p["functionResponse"], ensure_ascii=False)
        openai_messages.append({"role": role, "content": text or ""})

    # generation config -> OpenAI params
    gc = gemini_body.get("generationConfig", {})
    model_name = gemini_body.get("model", "")

    openai_body = {
        "model": GEMINI_MODEL_MAP.get(model_name, "deepseek-v4-flash"),
        "messages": openai_messages,
    }
    if "temperature" in gc:
        openai_body["temperature"] = gc["temperature"]
    if "maxOutputTokens" in gc:
        openai_body["max_tokens"] = gc["maxOutputTokens"]
    if "topP" in gc:
        openai_body["top_p"] = gc["topP"]
    if "stopSequences" in gc:
        openai_body["stop"] = gc["stopSequences"]

    log.debug("Gemini -> OpenAI converted body: %s", json.dumps(openai_body, ensure_ascii=False, indent=2))
    return openai_body


def _openai_to_gemini(openai_resp):
    """Convert OpenAI response body to Gemini-compatible format."""
    choices = openai_resp.get("choices", [])
    candidates = []
    for ch in choices:
        msg = ch.get("message", {})
        parts = []
        content = msg.get("content")
        if content:
            parts.append({"text": content})
        tc = msg.get("tool_calls")
        if tc:
            for t in tc:
                parts.append({"functionCall": {"name": t["function"]["name"], "args": json.loads(t["function"].get("arguments", "{}"))}})
        finish_map = {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP"}
        candidates.append({
            "content": {
                "role": "model",
                "parts": parts,
            },
            "finishReason": finish_map.get(ch.get("finish_reason", ""), "STOP"),
            "index": ch.get("index", 0),
        })

    gemini_resp = {"candidates": candidates}

    usage = openai_resp.get("usage")
    if usage:
        gemini_resp["usageMetadata"] = {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        }

    return gemini_resp


def _openai_chunk_to_gemini(line):
    """Convert a single OpenAI SSE line to Gemini SSE format."""
    if line.startswith("data: ") and line != "data: [DONE]":
        try:
            parsed = json.loads(line[6:])
        except Exception:
            return line
        choices = parsed.get("choices", [{}])
        candidates = []
        for ch in choices:
            delta = ch.get("delta", {})
            parts = []
            content = delta.get("content")
            if content:
                parts.append({"text": content})
            tc = delta.get("tool_calls")
            if tc:
                for t in tc:
                    fn = t.get("function", {})
                    parts.append({
                        "functionCall": {
                            "name": fn.get("name", ""),
                            "args": fn.get("arguments", "{}"),
                        }
                    })
            candidates.append({
                "content": {"parts": parts},
                "finishReason": ch.get("finish_reason", None),
                "index": ch.get("index", 0),
            })
        gemini_chunk = {"candidates": candidates}
        usage = parsed.get("usage")
        if usage:
            gemini_chunk["usageMetadata"] = {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            }
        return "data: " + json.dumps(gemini_chunk, ensure_ascii=False) + "\n\n"
    return line


# ──────────────────────────────
# Gemini API 路由
# ──────────────────────────────

@app.route("/v1beta/models/<model>:generateContent", methods=["POST"])
def gemini_generate(model):
    """Gemini non-streaming generateContent -> DeepSeek."""
    body = request.get_json(silent=True) or {}
    log.debug("=== Gemini generateContent (model=%s) ===", model)
    log.debug("Original body: %s", json.dumps(body, ensure_ascii=False, indent=2))

    openai_body = _gemini_to_openai(body)

    headers = {k: v for k, v in request.headers if k.lower() != "host"}
    resp = requests.post(
        f"{DEEPSEEK_BASE}/chat/completions",
        headers=headers, json=openai_body, stream=False
    )
    log.debug("DeepSeek response status: %d", resp.status_code)

    if resp.status_code != 200:
        log.warning("Upstream error (HTTP %d): %s", resp.status_code, resp.text)
        return _gemini_error(resp.status_code, resp.text)

    data = resp.json()
    gemini_resp = _openai_to_gemini(data)
    log.debug("Gemini response: %s", json.dumps(gemini_resp, ensure_ascii=False, indent=2))
    return Response(json.dumps(gemini_resp, ensure_ascii=False), content_type="application/json")


@app.route("/v1beta/models/<model>:streamGenerateContent", methods=["POST"])
def gemini_stream_generate(model):
    """Gemini streaming streamGenerateContent -> DeepSeek."""
    body = request.get_json(silent=True) or {}
    log.debug("=== Gemini streamGenerateContent (model=%s) ===", model)
    log.debug("Original body: %s", json.dumps(body, ensure_ascii=False, indent=2))

    openai_body = _gemini_to_openai(body)
    openai_body["stream"] = True

    headers = {k: v for k, v in request.headers if k.lower() != "host"}
    resp = requests.post(
        f"{DEEPSEEK_BASE}/chat/completions",
        headers=headers, json=openai_body, stream=True
    )
    log.debug("DeepSeek response status: %d", resp.status_code)

    if resp.status_code != 200:
        return _gemini_error(resp.status_code, resp.text)

    def generate():
        for chunk in resp.iter_lines():
            if not chunk:
                continue
            line = chunk.decode("utf-8", errors="replace")
            if line:
                yield _openai_chunk_to_gemini(line)

    return Response(
        stream_with_context(generate()),
        status=200,
        content_type="text/event-stream",
    )


# ──────────────────────────────
# OpenAI 兼容 - 模型列表
# ──────────────────────────────

@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
def list_models():
    """Return available models so Android Studio can validate the configuration."""
    models = [
        {"id": "deepseek-v4-flash", "object": "model", "created": 1700000000, "owned_by": "deepseek"},
        {"id": "deepseek-v4-pro", "object": "model", "created": 1700000000, "owned_by": "deepseek"},
        {"id": "deepseek-chat", "object": "model", "created": 1700000000, "owned_by": "deepseek"},
        {"id": "deepseek-reasoner", "object": "model", "created": 1700000000, "owned_by": "deepseek"},
    ]
    return Response(json.dumps({"object": "list", "data": models}), content_type="application/json")


def _gemini_error(status_code, text):
    """Return a Gemini-formatted error response."""
    try:
        detail = json.loads(text)
    except Exception:
        detail = {"message": text}
    return Response(json.dumps({
        "error": {
            "code": status_code,
            "message": detail.get("error", {}).get("message", detail.get("message", str(text))),
            "status": "UNAVAILABLE" if status_code >= 500 else "INVALID_ARGUMENT",
        }
    }), status=status_code, content_type="application/json")


@app.route("/chat/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    log.debug("=== REQUEST %s/chat/completions ===", DEEPSEEK_BASE)
    log.debug("Original body: %s", json.dumps(body, ensure_ascii=False, indent=2))

    # Only enable thinking for reasoner model
    model = body.get("model", "")
    if "reasoner" in model.lower():
        _enable_thinking(body)

    messages = body.get("messages", [])
    _inject_reasoning_content(messages)

    log.debug("Modified body: %s", json.dumps(body, ensure_ascii=False, indent=2))

    headers = {k: v for k, v in request.headers if k.lower() != "host"}
    streaming = body.get("stream", False)

    endpoint = f"{DEEPSEEK_BASE}/chat/completions"
    log.debug("Calling endpoint: %s", endpoint)
    resp = requests.post(
        endpoint,
        headers=headers, json=body, stream=True
    )
    log.debug("DeepSeek response status: %d", resp.status_code)

    # retry with thinking disabled on reasoning_content 400
    if resp.status_code == 400:
        err_body = resp.text
        if "reasoning_content" in err_body:
            log.warning("upstream rejected reasoning_content, retrying with thinking disabled")
            resp.close()
            _disable_thinking(body)
            resp = requests.post(
                endpoint,
                headers=headers, json=body, stream=True
            )
            log.debug("Retry response status: %d", resp.status_code)

    extra_headers = {}

    if resp.status_code != 200:
        error_body = resp.text
        log.warning("Upstream error (HTTP %d): %s", resp.status_code, error_body)
        return Response(error_body, status=resp.status_code,
                        content_type="application/json",
                        headers=extra_headers)

    def _extract_kv_cache(data_obj):
        usage = data_obj.get("usage") if isinstance(data_obj, dict) else None
        if usage:
            global upstream_kv_hit, upstream_kv_miss
            h = usage.get("prompt_cache_hit_tokens", 0)
            m = usage.get("prompt_cache_miss_tokens", 0)
            if h or m:
                upstream_kv_hit += h
                upstream_kv_miss += m
            return h, m
        return None, None

    if not streaming:
        try:
            data = resp.json()
        except Exception as e:
            log.warning("Failed to parse response JSON: %s", e)
            return Response(resp.text, status=resp.status_code,
                            headers=dict(resp.headers) | extra_headers)

        kv_hit, kv_miss = _extract_kv_cache(data)
        if kv_hit is not None:
            extra_headers["X-Upstream-KV"] = f"hit={kv_hit} miss={kv_miss}"

        msg = data.get("choices", [{}])[0].get("message", {})
        _cache_reasoning(msg)

        print(_fmt_stats(), flush=True)
        return Response(json.dumps(data), status=resp.status_code,
                        content_type="application/json",
                        headers=extra_headers)
    else:
        BUF_SIZE = 20

        def _flush(buf):
            if buf:
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": buf}, "index": 0}]
                }, ensure_ascii=False) + "\n\n"

        def generate():
            chunk_count = 0
            reasoning_acc = None
            content_acc = None
            tool_calls_acc = {}
            buf = ""
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk_count += 1
                    try:
                        parsed = json.loads(line[6:])
                        usage = parsed.get("usage")
                        if usage:
                            h = usage.get("prompt_cache_hit_tokens", 0)
                            m = usage.get("prompt_cache_miss_tokens", 0)
                            if h or m:
                                global upstream_kv_hit, upstream_kv_miss
                                upstream_kv_hit += h
                                upstream_kv_miss += m
                        delta = parsed.get("choices", [{}])[0].get("delta", {})
                        rc = delta.get("reasoning_content")
                        if rc:
                            reasoning_acc = (reasoning_acc or "") + rc
                        dc = delta.get("content")
                        if dc:
                            content_acc = (content_acc or "") + dc
                            buf += dc
                            if len(buf) >= BUF_SIZE:
                                yield from _flush(buf)
                                buf = ""
                            continue
                        tcs = delta.get("tool_calls")
                        if tcs:
                            for tc in tcs:
                                idx = tc.get("index")
                                if idx is not None:
                                    if idx not in tool_calls_acc:
                                        tool_calls_acc[idx] = {}
                                    for key in ("id", "type"):
                                        if key in tc:
                                            tool_calls_acc[idx][key] = tc[key]
                                    fn = tc.get("function")
                                    if fn:
                                        cur_fn = tool_calls_acc[idx].setdefault("function", {})
                                        for fk in ("name", "arguments"):
                                            if fk in fn:
                                                cur_fn[fk] = cur_fn.get(fk, "") + fn[fk]
                    except Exception as e:
                        log.warning("Stream parse error: %s", e)
                # non-content line: flush buffer first, then yield original line
                if buf:
                    yield from _flush(buf)
                    buf = ""
                yield line + "\n\n"
            if buf:
                yield from _flush(buf)

            log.debug("Stream ended, total chunks=%d", chunk_count)
            if reasoning_acc and tool_calls_acc:
                indices = sorted(tool_calls_acc.keys())
                ids = [tool_calls_acc[i].get("id", "") for i in indices]
                msg = {"content": content_acc or "", "tool_calls": [{"id": tid} for tid in ids], "reasoning_content": reasoning_acc}
                _cache_reasoning(msg)
            print(_fmt_stats(), flush=True)

        return Response(
            stream_with_context(generate()),
            status=resp.status_code,
            content_type="text/event-stream",
            headers={k: v for k, v in resp.headers.items()
                     if k.lower() not in ("content-length", "content-type")} | extra_headers
        )


@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_all(path):
    headers = {k: v for k, v in request.headers if k.lower() != "host"}
    body = request.get_json(silent=True) if request.method in ("POST", "PUT") else None
    endpoint = f"{DEEPSEEK_BASE}/{path}"
    log.debug("Calling endpoint: [%s] %s", request.method, endpoint)
    resp = requests.request(
        request.method, endpoint,
        headers=headers, json=body, stream=True
    )
    if resp.status_code != 200:
        error_body = resp.text
        log.warning("Upstream error (HTTP %d) on [%s] %s: %s",
                     resp.status_code, request.method, endpoint, error_body)
        return Response(error_body, status=resp.status_code,
                        content_type="application/json")
    return Response(resp.iter_content(), status=resp.status_code, headers=dict(resp.headers))


def _fmt_stats():
    rc_total = reasoning_cache_hits + reasoning_cache_misses
    rc_ratio = round(reasoning_cache_hits / rc_total, 4) if rc_total else 0
    kv_total = upstream_kv_hit + upstream_kv_miss
    kv_ratio = round(upstream_kv_hit / kv_total, 4) if kv_total else 0
    return (
        f"[reasoning_cache] entries={len(reasoning_cache)} hits={reasoning_cache_hits} misses={reasoning_cache_misses} ratio={rc_ratio} "
        f"[kv_cache] hit={upstream_kv_hit} miss={upstream_kv_miss} ratio={kv_ratio}"
    )


@app.route("/stats", methods=["GET"])
def stats():
    rc_total = reasoning_cache_hits + reasoning_cache_misses
    rc_ratio = round(reasoning_cache_hits / rc_total, 4) if rc_total else 0
    kv_total = upstream_kv_hit + upstream_kv_miss
    kv_ratio = round(upstream_kv_hit / kv_total, 4) if kv_total else 0
    return Response(json.dumps({
        "reasoning_cache": {
            "entries": len(reasoning_cache),
            "hits": reasoning_cache_hits,
            "misses": reasoning_cache_misses,
            "hit_ratio": rc_ratio,
        },
        "upstream_kv_cache": {
            "hit_tokens": upstream_kv_hit,
            "miss_tokens": upstream_kv_miss,
            "hit_ratio": kv_ratio,
        },
    }, ensure_ascii=False), content_type="application/json")


@app.route("/", methods=["GET", "POST", "PUT", "DELETE"])
def root():
    return proxy_all("")


def _periodic_stats():
    while True:
        threading.Event().wait(30)
        print(_fmt_stats(), flush=True)


if __name__ == "__main__":
    print(f"Log file: {LOG_FILE}", flush=True)
    threading.Thread(target=_periodic_stats, daemon=True).start()
    print(_fmt_stats(), flush=True)
    log.info("Listening on 127.0.0.2:8081")
    app.run(host="127.0.0.2", port=8081)