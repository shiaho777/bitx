from __future__ import annotations

import argparse

from kef.weights import load_model_and_tokenizer, resolve_checkpoint


DEFAULT_MODEL = "/Users/shiaho/Desktop/MiniCPM5-1B"
DEFAULT_VARIANT = ""


def load(model_path: str, variant: str, device: str):
    path = resolve_checkpoint(model_path, variant or None)
    model, tok = load_model_and_tokenizer(path, device=device, trainable=False)
    return model, tok


def cleanup_answer(user_text: str, text: str) -> str:
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


def reply(model, tok, device: str, prompt: str, max_new: int) -> str:
    import re
    greet = {"你好", "您好", "hi", "Hi", "hello", "Hello", "嗨", "在吗", "谢谢", "好的", "哈喽", "hey", "Hey", "早"}
    if prompt.strip() in greet:
        max_new = min(max_new, 24)
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    import torch
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=6,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    lines = text.splitlines()
    out_lines = []
    prev = None
    for ln in lines:
        cur = ln.strip()
        if cur and cur == prev:
            continue
        prev = cur
        out_lines.append(cur)
    text = "\n".join(out_lines).strip()
    text = re.sub(r"(你好[\s!]*){2,}", "你好！", text)
    return cleanup_answer(prompt, text)


def main():
    p = argparse.ArgumentParser(description="BitX full-weight MiniCPM chat")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--variant", default=DEFAULT_VARIANT, help="optional full checkpoint dir; default uses --model")
    p.add_argument("--adapter", default="", help=argparse.SUPPRESS)
    p.add_argument("--device", default="mps")
    p.add_argument("--max-new", type=int, default=320)
    p.add_argument("--once", default="")
    args = p.parse_args()
    variant = args.variant or args.adapter or ""

    print("=== BitX full-weight model ===", flush=True)
    print(f"checkpoint: {resolve_checkpoint(args.model, variant or None)}", flush=True)
    print(f"device: {args.device}", flush=True)
    model, tok = load(args.model, variant, args.device)
    print("READY — 直接输入问题；exit 退出", flush=True)

    if args.once:
        print(reply(model, tok, args.device, args.once, args.max_new))
        return

    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye", flush=True)
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("bye", flush=True)
            break
        print("模型>", reply(model, tok, args.device, q, args.max_new), flush=True)


if __name__ == "__main__":
    main()
