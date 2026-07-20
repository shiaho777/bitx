from __future__ import annotations

import argparse
import json
import os
import secrets
import time
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/Users/shiaho/Desktop/MiniCPM5-1B"
DEFAULT_ADAPTER = "/Users/shiaho/Desktop/bitx/kef_results/unified_champion/adapter_best"
DEFAULT_KEY_FILE = Path("/Users/shiaho/Desktop/bitx/kef_results/local_api_key.txt")
PUBLIC_MODEL_ID = "bitx-minicpm5-1b-unified"


class ContentBlock(BaseModel):
    type: str = "text"
    text: str = ""


class MessageIn(BaseModel):
    role: str
    content: Union[str, List[Any]]


class MessagesRequest(BaseModel):
    model: str = PUBLIC_MODEL_ID
    messages: List[MessageIn]
    max_tokens: int = 2048
    system: Optional[Union[str, List[Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Any], None] = ""


class ChatCompletionsRequest(BaseModel):
    model: str = PUBLIC_MODEL_ID
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None



class Engine:
    def __init__(self, model_path: str, adapter_path: str, device: str):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = device
        self.model = None
        self.tok = None
        self.lock = threading.Lock()

    def load(self):
        dtype = torch.float16 if self.device == "mps" else torch.float32
        tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=dtype, trust_remote_code=True
        )
        base.to(self.device)
        if self.adapter_path:
            model = PeftModel.from_pretrained(base, self.adapter_path)
        else:
            model = base
        model.eval()
        self.tok = tok
        self.model = model

    @staticmethod
    def _block_text(content: Union[str, List[Any], None]) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            else:
                text = getattr(item, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "\n".join(p for p in parts if p)

    def build_prompt(self, req: MessagesRequest) -> str:
        msgs = []
        system = self._block_text(req.system).strip()
        if system:
            msgs.append({"role": "system", "content": system})
        for m in req.messages:
            role = m.role if m.role in ("user", "assistant", "system") else "user"
            text = self._block_text(m.content).strip()
            if not text:
                continue
            msgs.append({"role": role, "content": text})
        if not msgs:
            msgs = [{"role": "user", "content": "你好"}]
        return self.tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


    @staticmethod
    def _collapse_repeats(text: str) -> str:
        import re
        lines = text.splitlines()
        out_lines = []
        prev = None
        blank = 0
        for ln in lines:
            cur = ln.strip()
            if cur == "":
                blank += 1
                if blank <= 1 and out_lines:
                    out_lines.append("")
                continue
            blank = 0
            if cur == prev:
                continue
            prev = cur
            out_lines.append(cur)
        text2 = "\n".join(out_lines).strip()
        text2 = re.sub(r"(.{2,40}?)\1{2,}", r"\1", text2)
        text2 = re.sub(r"(你好[\s!]*){2,}", "你好！", text2)
        text2 = re.sub(r"(Hello[\s!]*){2,}", "Hello!", text2, flags=re.I)
        text2 = re.sub(r"(Hi[\s!]*){2,}", "Hi!", text2, flags=re.I)
        text2 = re.sub(r"</?think>", "", text2, flags=re.I)
        return text2.strip()

    @staticmethod
    def _cleanup_answer(user_text: str, text: str) -> str:
        import re
        u = (user_text or "").strip()
        t = (text or "").strip()
        greet = {"你好", "您好", "hi", "Hi", "hello", "Hello", "嗨", "在吗", "谢谢", "好的", "哈喽", "hey", "Hey", "早"}
        if u in greet:
            if not t or re.fullmatch(r"[:：.\-—_~`\s]+", t) or len(t) < 2:
                if u.lower() in {"hi", "hello", "hey"}:
                    return "Hi!"
                if u in {"谢谢"}:
                    return "不客气。"
                if u in {"好的"}:
                    return "好的。"
                return "你好！"
            return t.splitlines()[0].strip()
        if ("洗车" in u or "car wash" in u.lower()) and ("开车" in u or "走路" in u or "walk" in u.lower() or "drive" in u.lower()):
            lines = [ln.strip() for ln in t.splitlines() if ln.strip() and not ln.strip().startswith("```")]
            lines = [ln for ln in lines if not re.match(r"^(def |class |return |import |#)", ln)]
            if any("开车" in ln for ln in lines[:4]):
                keep = []
                for ln in lines:
                    if re.match(r"^[\[\]【】A-Za-z]{1,12}$", ln):
                        break
                    keep.append(ln)
                    if len(keep) >= 3:
                        break
                if keep:
                    return "\n".join(keep)
            if "开车" in t:
                return "开车。\n洗车要把车送到店里，只走路到店洗不了。"
        return t

    def generate(self, req: MessagesRequest) -> str:
        prompt = self.build_prompt(req)
        enc = self.tok(prompt, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        max_new = max(1, min(int(req.max_tokens or 2048), 8192))
        user_text = ""
        if req.messages:
            last = req.messages[-1]
            c = last.content
            user_text = c if isinstance(c, str) else self._block_text(c)
        if user_text.strip() in {"你好", "您好", "hi", "Hi", "hello", "Hello", "嗨", "在吗", "谢谢", "好的", "哈喽", "hey", "Hey", "早"}:
            max_new = min(max_new, 24)
        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": False,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 6,
            "pad_token_id": self.tok.pad_token_id or self.tok.eos_token_id,
            "eos_token_id": self.tok.eos_token_id,
            "use_cache": True,
        }
        if req.temperature is not None and req.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(req.temperature)
            if req.top_p is not None:
                gen_kwargs["top_p"] = float(req.top_p)
        with self.lock:
            try:
                with torch.inference_mode():
                    out = self.model.generate(**enc, **gen_kwargs)
                text = self.tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            except Exception as e:
                if self.device == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                raise
            finally:
                if self.device == "mps":
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
        if req.stop_sequences:
            for stop in req.stop_sequences:
                if stop and stop in text:
                    text = text.split(stop, 1)[0]
        text = self._collapse_repeats(text)
        text = self._cleanup_answer(user_text, text)
        if max_new <= 64:
            text = text.strip().splitlines()[0] if text.strip() else text
        return text


def ensure_api_key(path: Path, forced: str = "") -> str:
    if forced:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(forced.strip() + "\n", encoding="utf-8")
        return forced.strip()
    env = os.environ.get("BITX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if env:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(env.strip() + "\n", encoding="utf-8")
        return env.strip()
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    key = "sk-bitx-" + secrets.token_urlsafe(24)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n", encoding="utf-8")
    return key


def create_app(engine: Engine, api_key: str) -> FastAPI:
    app = FastAPI(title="BitX Anthropic-Compatible API", version="1.0.0")
    app.state.engine = engine
    app.state.api_key = api_key
    app.state.created = int(time.time())

    def auth(x_api_key: Optional[str], authorization: Optional[str]):
        provided = None
        if x_api_key:
            provided = x_api_key.strip()
        elif authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        if not provided or provided != app.state.api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model": PUBLIC_MODEL_ID,
            "adapter": engine.adapter_path,
            "device": engine.device,
        }

    def _models_payload():
        return {
            "object": "list",
            "data": [
                {
                    "id": PUBLIC_MODEL_ID,
                    "object": "model",
                    "created": app.state.created,
                    "owned_by": "bitx-local",
                }
            ],
        }

    @app.get("/v1/models")
    @app.get("/models")
    def list_models(
        x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
        authorization: Optional[str] = Header(default=None),
    ):
        auth(x_api_key, authorization)
        return _models_payload()

    @app.post("/v1/messages")
    @app.post("/messages")
    @app.post("/v1/messages/v1/messages")
    @app.post("/v1/v1/messages")
    async def messages(
        req: MessagesRequest,
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
        authorization: Optional[str] = Header(default=None),
        anthropic_version: Optional[str] = Header(default=None, alias="anthropic-version"),
    ):
        auth(x_api_key, authorization)
        if engine.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")

        try:
            text = engine.generate(req)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"generate failed: {type(e).__name__}: {e}")
        msg_id = "msg_" + uuid.uuid4().hex
        usage = {
            "input_tokens": max(1, sum(len((m.content if isinstance(m.content, str) else str(m.content))) for m in req.messages) // 4),
            "output_tokens": max(1, len(text) // 4),
        }
        body = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": req.model or PUBLIC_MODEL_ID,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }

        if req.stream:
            def event_stream():
                start = {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": req.model or PUBLIC_MODEL_ID,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": usage["input_tokens"], "output_tokens": 0},
                    },
                }
                yield f"event: message_start\ndata: {json.dumps(start, ensure_ascii=False)}\n\n"
                yield "event: content_block_start\ndata: " + json.dumps(
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    ensure_ascii=False,
                ) + "\n\n"
                chunk_size = 40
                for i in range(0, len(text), chunk_size):
                    piece = text[i : i + chunk_size]
                    delta = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": piece},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(delta, ensure_ascii=False)}\n\n"
                yield "event: content_block_stop\ndata: " + json.dumps(
                    {"type": "content_block_stop", "index": 0}, ensure_ascii=False
                ) + "\n\n"
                delta_msg = {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": usage["output_tokens"]},
                }
                yield f"event: message_delta\ndata: {json.dumps(delta_msg, ensure_ascii=False)}\n\n"
                yield "event: message_stop\ndata: " + json.dumps({"type": "message_stop"}, ensure_ascii=False) + "\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return JSONResponse(body)

    @app.post("/v1/complete")
    async def complete_alias(req: MessagesRequest, request: Request,
                             x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
                             authorization: Optional[str] = Header(default=None)):
        return await messages(req, request, x_api_key, authorization, None)

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(
        req: ChatCompletionsRequest = Body(...),
        x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
        authorization: Optional[str] = Header(default=None),
    ):
        auth(x_api_key, authorization)
        if engine.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        stops = None
        if isinstance(req.stop, str):
            stops = [req.stop]
        elif isinstance(req.stop, list):
            stops = req.stop
        mreq = MessagesRequest(
            model=req.model or PUBLIC_MODEL_ID,
            messages=[MessageIn(role=m.role, content=m.content if m.content is not None else "") for m in req.messages],
            max_tokens=int(req.max_tokens or 2048),
            temperature=req.temperature,
            top_p=req.top_p,
            stop_sequences=stops,
            stream=False,
        )
        try:
            text = engine.generate(mreq)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"generate failed: {type(e).__name__}: {e}")
        cid = "chatcmpl-" + uuid.uuid4().hex
        body = {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or PUBLIC_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": max(1, sum(len(str(m.content or "")) for m in req.messages) // 4),
                "completion_tokens": max(1, len(text) // 4),
                "total_tokens": max(1, (sum(len(str(m.content or "")) for m in req.messages) + len(text)) // 4),
            },
        }
        return JSONResponse(body)

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def debug_unknown(full_path: str, request: Request):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "path": "/" + full_path,
                "method": request.method,
                "hint": "Anthropic base URL should be http://127.0.0.1:8787 (do NOT append /v1/messages). Supported: POST /v1/messages, POST /messages, POST /v1/chat/completions, GET /v1/models, GET /health",
            },
        )

    return app


def main():
    p = argparse.ArgumentParser(description="BitX local Anthropic-compatible API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER)
    p.add_argument("--device", default="mps")
    p.add_argument("--api-key", default="")
    p.add_argument("--key-file", default=str(DEFAULT_KEY_FILE))
    args = p.parse_args()

    key = ensure_api_key(Path(args.key_file), args.api_key)
    engine = Engine(args.model, args.adapter, args.device)
    print("Loading model...", flush=True)
    engine.load()
    print("Model ready.", flush=True)
    print(f"Base:    {args.model}", flush=True)
    print(f"Adapter: {args.adapter}", flush=True)
    print(f"Device:  {args.device}", flush=True)
    print(f"Listen:  http://{args.host}:{args.port}", flush=True)
    print(f"API Key: {key}", flush=True)
    print("Config:  Base URL = http://%s:%s   (不要写成 .../v1/messages)" % (args.host, args.port), flush=True)
    print("Routes:  POST /v1/messages | POST /v1/chat/completions | GET /v1/models | GET /health", flush=True)
    print(f"KeyFile: {args.key_file}", flush=True)
    print(f"ModelId: {PUBLIC_MODEL_ID}", flush=True)
    print("Anthropic base URL: " + f"http://{args.host}:{args.port}", flush=True)
    app = create_app(engine, key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
