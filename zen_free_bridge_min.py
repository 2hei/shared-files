#!/usr/bin/env python3
"""zen_free_bridge_min.py — opencode zen免费模型的最小桥接。

前提：本机已运行 opencode serve --port 4199 --hostname 127.0.0.1
用途：把opencode的session API翻译成OpenAI兼容接口
启动：python zen_free_bridge_min.py   （监听127.0.0.1:4200）
验证：curl http://127.0.0.1:4200/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"glm-5.3-flash-free","messages":[{"role":"user","content":"ping"}]}'
"""
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

OPENCODE_URL = "http://127.0.0.1:4199"
BRIDGE_PORT = 4200
TIMEOUT = 600.0

app = FastAPI()
client = httpx.AsyncClient(timeout=TIMEOUT)


def split_model(model: str) -> tuple[str, str]:
    if "/" in model:
        provider, model_id = model.split("/", 1)
        return provider, model_id
    return "opencode", model  # 裸模型名默认走opencode


def flatten_messages(messages: list) -> tuple[str, str]:
    system_parts, text_parts = [], []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(
                str(p.get("text") or "") for p in content
                if isinstance(p, dict) and p.get("type") in ("text", "input_text")
            )
        content = str(content or "").strip()
        if not content:
            continue
        (system_parts if msg.get("role") == "system" else text_parts).append(content)
    return "\n\n".join(system_parts), "\n\n".join(text_parts)


def openai_error(message: str, err_type: str) -> dict:
    return {"error": {"message": message, "type": err_type, "code": None, "param": None}}


async def call_opencode(provider: str, model_id: str, system: str, text: str) -> dict:
    created = await client.post(f"{OPENCODE_URL}/session", json={"title": "zen-free-bridge"})
    created.raise_for_status()
    session_id = created.json()["id"]
    body = {
        "parts": [{"type": "text", "text": text}],
        "model": {"providerID": provider, "modelID": model_id},
    }
    if system:
        body["system"] = system
    try:
        resp = await client.post(f"{OPENCODE_URL}/session/{session_id}/message", json=body)
        resp.raise_for_status()
        return resp.json()
    finally:
        try:
            await client.delete(f"{OPENCODE_URL}/session/{session_id}")
        except httpx.HTTPError:
            pass


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    model = str(payload.get("model") or "")
    provider, model_id = split_model(model)
    system, text = flatten_messages(payload.get("messages"))
    if not model_id or not text:
        return JSONResponse(status_code=400, content=openai_error(
            "both 'model' and 'messages' are required", "invalid_request_error"))
    try:
        result = await call_opencode(provider, model_id, system, text)
    except httpx.HTTPStatusError as e:
        status = 429 if e.response.status_code == 429 else 502
        return JSONResponse(status_code=status, content=openai_error(
            f"opencode serve {e.response.status_code}: {e.response.text[:500]}", "upstream_error"))
    except httpx.HTTPError as e:
        return JSONResponse(status_code=502, content=openai_error(
            f"opencode serve unavailable: {e}", "upstream_error"))

    info = result.get("info") or {}
    if info.get("error"):
        return JSONResponse(status_code=502, content=openai_error(str(info["error"]), "upstream_error"))
    content = "".join(
        str(part.get("text") or "")
        for part in result.get("parts") or []
        if part.get("type") == "text"
    )
    tokens = info.get("tokens") or {}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": int(tokens.get("input") or 0),
            "completion_tokens": int(tokens.get("output") or 0),
            "total_tokens": int(tokens.get("total") or 0),
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=BRIDGE_PORT, log_level="warning")

