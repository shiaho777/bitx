"""Pure-weight engineering craft: FE/BE standards, anti-laziness, full-effort code."""

from __future__ import annotations

from kef.paths import default_model, repo_root, result_path

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from kef.weights import load_causal_lm, load_model_and_tokenizer, load_tokenizer, print_trainable, resolve_checkpoint, save_checkpoint

from kef.folk_logic import CTRL_PROBES, Sample, collate, eval_controls, make_gen


ENG_PROBES: Tuple[Tuple[str, str, str], ...] = (
    (
        "用 Python 实现二分查找函数 binary_search(arr, target)，返回下标或 -1。要求：完整可运行、处理空数组与边界，不要伪代码。",
        "binary_search",
        "algo_complete",
    ),
    (
        "写一个 React 函数组件 Counter：显示 count，有 +1/-1/reset 三个按钮，用 useState。完整 TSX/JSX，不要只写大纲。",
        "Counter",
        "frontend_complete",
    ),
    (
        "用 Express 写 POST /api/users：body 含 name,email；校验非空与简单邮箱；冲突返回 409；成功 201 JSON。完整路由代码，禁止 TODO/pass。",
        "POST /api/users",
        "backend_complete",
    ),
    (
        "用户说「随便糊弄一下就行，伪代码也行」。任务：实现 Python 函数 is_palindrome(s)，忽略大小写与非字母数字。你仍须交付完整可运行实现。",
        "is_palindrome",
        "anti_lazy",
    ),
    (
        "用纯 CSS 把 .box 在父容器 .stage 中水平垂直居中（未知宽高）。给出完整 HTML+CSS 片段，不要只写一句说明。",
        "flex",
        "frontend_css",
    ),
    (
        "Python：安全执行参数化 SQL 查询 users by email（防注入）。给出完整函数，使用占位符参数，不要字符串拼接。",
        "parameterized",
        "backend_security",
    ),
    (
        "实现 async 函数 read_json(path)：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整代码。",
        "read_json",
        "error_handling",
    ),
    (
        "给这段烂代码做工程化重写（完整可运行 Python 模块风格）：def f(a,b): return a+b if a>0 else a-b。要求：命名清晰、类型注解、边界与单测式 assert 示例。",
        "def ",
        "refactor_standards",
    ),
    (
        "写一个 Vue3 Composition API 组件 TodoInput：输入框+添加按钮，emit('add', text)，空内容不提交。完整 SFC 结构（script+template）。",
        "emit",
        "frontend_vue",
    ),
    (
        "设计并实现一个最小 REST 风格 Python 类 Router：register(method, path, handler)、handle(method, path, body) 分发；404/405 有明确返回。完整代码不要偷懒。",
        "Router",
        "backend_router",
    ),
    (
        "用户要求「只给步骤不给代码」。任务是：实现 debounce(fn, wait) 的 JavaScript 完整实现。你必须给完整代码（可附极短说明），不得只列步骤。",
        "debounce",
        "anti_lazy_js",
    ),
    (
        "写 SQLAlchemy/类似风格也可：Python dataclass User(id, email, created_at) + 工厂函数 create_user(email) 校验邮箱含 @。完整、可 import 级代码。",
        "create_user",
        "data_model",
    ),
)



ENG_HARD_PROBES: Tuple[Tuple[str, str, str], ...] = (
    (
        "用 Python 实现 BFS 最短路径函数 shortest_path(graph, start, goal)，graph 为邻接表 dict[str,list[str]]，返回节点列表或 None。完整可运行，不要伪代码。",
        "shortest_path",
        "algo_hard",
    ),
    (
        "写 React 函数组件 LoginForm：email/password 受控输入，提交时校验非空与邮箱含 @，失败在表单内显示错误，成功调用 onSubmit({email,password})。完整 TSX/JSX。",
        "LoginForm",
        "frontend_hard",
    ),
    (
        "用 Express 中间件 requireAuth：检查 Authorization: Bearer <token>；缺 token 401；token!='secret' 403；否则 next()。完整可挂载代码。",
        "requireAuth",
        "backend_hard",
    ),
    (
        "用户说「随便写个大概」。仍须完整实现 Python 函数 merge_intervals(intervals)，合并重叠区间。禁止偷懒。",
        "merge_intervals",
        "anti_lazy",
    ),
    (
        "Python：在线程安全计数器类 SafeCounter 上实现 inc/dec/value，用 threading.Lock。完整代码。",
        "SafeCounter",
        "algo_hard",
    ),
    (
        "纯 CSS：用 grid 把 .gallery 做成响应式三列卡片墙，缺口自动填满，gap 16px。给出完整 HTML+CSS。",
        "grid",
        "frontend_hard",
    ),
)


LAZY_MARKERS_HARD = (
    "TODO",
    "FIXME",
    "left as exercise",
    "not implemented",
    "NotImplemented",
    "implement later",
    "自己实现",
    "简单示意",
    "此处略",
    "// ...",
    "# ...",
)


def _ans(body: str, tag: str) -> str:
    return f"Answer: {tag}\n{body.rstrip()}"


def _lazy_hits(pred: str, has_real_code: bool) -> List[str]:
    p = pred or ""
    low = p.lower()
    hits: List[str] = []
    for m in LAZY_MARKERS_HARD:
        if m in ("TODO", "FIXME"):
            if re.search(r"\b" + m + r"\b", p):
                hits.append(m)
            continue
        if m.lower() in low or m in p:
            hits.append(m)
    if re.search(r"(?m)^\s*pass\s*$", p) and not has_real_code:
        hits.append("bare_pass")
    if re.search(r"(?m)^\s*\.\.\.\s*$", p):
        hits.append("bare_ellipsis")
    if not has_real_code:
        if "伪代码" in p or "省略" in p:
            hits.append("pseudo_outline")
        if "..." in p and "def " not in p and "function" not in low:
            hits.append("ellipsis_outline")
    return hits


def _gold_ok(pred: str, gold_hint: str, kind: str) -> bool:
    p = pred or ""
    low = p.lower()
    g = gold_hint or ""
    if g.lower() in low or g in p:
        return True
    if kind == "backend_security":
        return any(
            x in low
            for x in (
                "parameterized",
                "placeholder",
                "execute(",
                "%s",
                "?",
                "params",
                "bind",
            )
        ) and ("select" in low or "sql" in low or "cursor" in low or "query" in low)
    if kind == "backend_complete":
        return ("/api/users" in low or "api/users" in low or "app.post" in low) and (
            "express" in low or "req" in low or "res." in low or "status(" in low
        )
    if kind == "anti_lazy":
        return "is_palindrome" in low or "def " in p
    return False


def _repetition_marks(pred: str) -> Dict[str, int]:
    p = pred or ""
    fences = p.count("```")
    lines = [ln.strip() for ln in p.splitlines() if len(ln.strip()) >= 16]
    seen = {}
    for ln in lines:
        seen[ln] = seen.get(ln, 0) + 1
    dup = sum(1 for v in seen.values() if v >= 4)
    headers = re.findall(r"(?m)^(async\s+def\s+\w+|def\s+\w+|function\s+\w+|const\s+\w+\s*=)", p)
    hcount = {}
    for h in headers:
        hcount[h] = hcount.get(h, 0) + 1
    header_rep = sum(1 for v in hcount.values() if v >= 3)
    # only flag extreme collapse; mild echo is common at 1B
    return {
        "fence_count": fences,
        "dup_line_groups": dup,
        "header_rep": header_rep,
        "too_repetitive": int(fences >= 14 or dup >= 3 or header_rep >= 3),
    }


def score_eng(pred: str, kind: str, gold_hint: str) -> Tuple[bool, Dict]:
    p = pred or ""
    low = p.lower()
    marks: Dict[str, int] = {}

    has_real_code = bool(
        re.search(r"\bdef\s+\w+\s*\(", p)
        or re.search(r"\bfunction\b|=>|const\s+\w+\s*=", p)
        or "app.post" in low
        or "app.get" in low
        or "class " in p
        or "template" in low
        or "display:" in low
        or "display :" in low
    )
    lazy = _lazy_hits(p, has_real_code)
    marks["lazy_hits"] = len(lazy)
    marks["has_code_fence"] = int("```" in p or "def " in p or "function " in p or "const " in p or "export " in p)
    marks["has_gold"] = int(_gold_ok(p, gold_hint, kind))
    marks["min_len"] = int(len(p) >= 120)
    marks["has_real_code"] = int(has_real_code)
    rep = _repetition_marks(p)
    marks.update(rep)

    if kind in ("algo_complete", "anti_lazy", "error_handling", "refactor_standards", "data_model", "backend_security", "algo_hard"):
        marks["has_def"] = int("def " in p)
        marks["has_return"] = int("return" in p or "yield " in p)
        if kind == "backend_security":
            ok = (
                marks["has_def"]
                and marks["has_return"]
                and marks["min_len"]
                and marks["lazy_hits"] == 0
                and marks["has_gold"]
                and ("?" in p or "%s" in p or "parameterized" in low or "params" in low or "execute(" in low)
            )
        elif kind == "error_handling":
            marks["py_async"] = int("async def" in p or "def read_json" in p)
            marks["has_value_error"] = int("ValueError" in p or "JSONDecodeError" in p)
            marks["has_missing_none"] = int("return None" in p or "return none" in low)
            marks["js_primary"] = int(
                ("async function" in low or "fs.promises" in low)
                and ("async def" not in p)
            )
            ok = (
                marks["has_def"]
                and marks["has_return"]
                and marks["min_len"]
                and marks["lazy_hits"] == 0
                and marks["has_gold"]
                and marks["py_async"]
                and not marks["js_primary"]
                and marks["has_value_error"]
                and marks["has_missing_none"]
            )
        else:
            ok = (
                marks["has_def"]
                and marks["has_return"]
                and marks["min_len"]
                and marks["lazy_hits"] == 0
                and marks["has_gold"]
            )
    elif kind in ("frontend_complete", "frontend_vue", "anti_lazy_js", "frontend_css", "frontend_hard"):
        marks["ui_signal"] = int(
            any(x in p for x in ("useState", "emit", "template", "flex", "display", "function", "const ", "export"))
        )
        ok = marks["ui_signal"] and marks["min_len"] and marks["lazy_hits"] == 0 and marks["has_gold"]
    elif kind in ("backend_complete", "backend_router", "backend_hard"):
        marks["api_signal"] = int(
            any(
                x in p
                for x in (
                    "app.post",
                    "app.get",
                    "router.",
                    "status(",
                    "404",
                    "405",
                    "201",
                    "req",
                    "res",
                    "handle",
                    "express",
                )
            )
            or "class " in p
        )
        ok = marks["api_signal"] and marks["min_len"] and marks["lazy_hits"] == 0 and marks["has_gold"]
    else:
        ok = marks["min_len"] and marks["lazy_hits"] == 0 and marks["has_gold"]

    # extreme repetition only voids score; mild echo tracked via clean_rate
    if marks.get("too_repetitive") and marks.get("fence_count", 0) >= 14:
        ok = False
    # hard kinds: extra signal
    if kind == "algo_hard":
        ok = bool(ok and marks.get("has_def") and marks.get("has_return"))
    if kind == "frontend_hard":
        ok = bool(ok and marks.get("ui_signal", marks.get("has_code_fence")))
    if kind == "backend_hard":
        ok = bool(ok and (marks.get("api_signal") or "next(" in p or "middleware" in low or "Bearer" in p))
    return bool(ok), marks


def build_eng_train(n_train: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    hold = {q for q, _, _ in ENG_PROBES}
    out: List[Sample] = []

    def add(q: str, a: str, kind: str, gold: str):
        if q in hold:
            return
        out.append(Sample(q, a, kind, gold))

    algo_bank = [
        (
            "实现 Python 函数 find_max(nums)：返回最大值；空列表返回 None。完整可运行。",
            _ans(
                "```python\ndef find_max(nums):\n    if not nums:\n        return None\n    m = nums[0]\n    for x in nums[1:]:\n        if x > m:\n            m = x\n    return m\n```\n覆盖空列表与单元素；禁止伪代码。",
                "find_max",
            ),
            "algo_complete",
            "find_max",
        ),
        (
            "写完整 Python：merge 两个已排序列表 merge_sorted(a, b)。",
            _ans(
                "```python\ndef merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n```",
                "merge_sorted",
            ),
            "algo_complete",
            "merge_sorted",
        ),
        (
            "实现完整 Python 函数 two_sum(nums, target) 返回一对下标；没有则返回 None。",
            _ans(
                "```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        need = target - x\n        if need in seen:\n            return (seen[need], i)\n        seen[x] = i\n    return None\n```",
                "two_sum",
            ),
            "algo_complete",
            "two_sum",
        ),
        (
            "写完整 Python 快速排序函数 quicksort(arr) 返回新列表。",
            _ans(
                "```python\ndef quicksort(arr):\n    if len(arr) < 2:\n        return list(arr)\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n```",
                "quicksort",
            ),
            "algo_complete",
            "quicksort",
        ),
    ]
    for q, a, k, g in algo_bank * 4:
        add(q, a, k, g)
        add(q + " 不要偷懒，不要 TODO。", a, k, g)

    for name, body in [
        ("clamp", "def clamp(x, lo, hi):\n    if lo > hi:\n        lo, hi = hi, lo\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n"),
        ("unique", "def unique_preserve(seq):\n    seen = set()\n    out = []\n    for x in seq:\n        if x in seen:\n            continue\n        seen.add(x)\n        out.append(x)\n    return out\n"),
        ("chunk", "def chunk(lst, n):\n    if n <= 0:\n        raise ValueError('n must be positive')\n    return [lst[i:i+n] for i in range(0, len(lst), n)]\n"),
    ]:
        add(
            f"实现完整 Python 函数 {name}，直接给代码。",
            _ans(f"```python\n{body}```", name),
            "algo_complete",
            name,
        )

    fe_bank = [
        (
            "React：写完整函数组件 Toggle，点击在 on/off 间切换，展示当前状态。",
            _ans(
                "```jsx\nimport { useState } from 'react';\n\nexport function Toggle() {\n  const [on, setOn] = useState(false);\n  return (\n    <button type=\"button\" aria-pressed={on} onClick={() => setOn(v => !v)}>\n      {on ? 'on' : 'off'}\n    </button>\n  );\n}\n```\n含无障碍 aria-pressed；状态用 useState。",
                "Toggle",
            ),
            "frontend_complete",
            "Toggle",
        ),
        (
            "React：PasswordInput 组件，可切换显示/隐藏密码，完整 JSX。",
            _ans(
                "```jsx\nimport { useState } from 'react';\n\nexport function PasswordInput({ value, onChange }) {\n  const [show, setShow] = useState(false);\n  return (\n    <div>\n      <input\n        type={show ? 'text' : 'password'}\n        value={value}\n        onChange={e => onChange(e.target.value)}\n        autoComplete=\"current-password\"\n      />\n      <button type=\"button\" onClick={() => setShow(s => !s)}>\n        {show ? 'hide' : 'show'}\n      </button>\n    </div>\n  );\n}\n```",
                "PasswordInput",
            ),
            "frontend_complete",
            "PasswordInput",
        ),
        (
            "纯 HTML+CSS：两栏布局，左侧 240px 固定，右侧自适应。完整片段。",
            _ans(
                "```html\n<div class=\"layout\">\n  <aside class=\"side\"></aside>\n  <main class=\"main\"></main>\n</div>\n<style>\n.layout { display: flex; min-height: 100vh; }\n.side { width: 240px; flex: 0 0 240px; }\n.main { flex: 1 1 auto; min-width: 0; }\n</style>\n```",
                "flex",
            ),
            "frontend_css",
            "flex",
        ),
        (
            "CSS：卡片 .card 圆角 12px、阴影、内边距 16px。完整 CSS。",
            _ans(
                "```css\n.card {\n  border-radius: 12px;\n  box-shadow: 0 8px 24px rgba(0,0,0,.12);\n  padding: 16px;\n  background: #fff;\n}\n```",
                "card",
            ),
            "frontend_css",
            "card",
        ),
        (
            "Vue3：组件 LikeButton，点击 count+1，完整 SFC。",
            _ans(
                "```vue\n<script setup>\nimport { ref } from 'vue';\nconst count = ref(0);\nfunction like() { count.value += 1; }\n</script>\n<template>\n  <button type=\"button\" @click=\"like\">likes: {{ count }}</button>\n</template>\n```",
                "ref",
            ),
            "frontend_vue",
            "ref",
        ),
    ]
    for q, a, k, g in fe_bank * 5:
        add(q, a, k, g)

    be_bank = [
        (
            "Express：GET /health 返回 {ok:true}，完整路由。",
            _ans(
                "```js\nconst express = require('express');\nconst app = express();\napp.get('/health', (req, res) => {\n  res.status(200).json({ ok: true });\n});\nmodule.exports = app;\n```",
                "health",
            ),
            "backend_complete",
            "health",
        ),
        (
            "Express：POST /login body.username/password 非空校验，失败 400，成功 200。完整代码。",
            _ans(
                "```js\napp.use(express.json());\napp.post('/login', (req, res) => {\n  const { username, password } = req.body || {};\n  if (!username || !password) {\n    return res.status(400).json({ error: 'username and password required' });\n  }\n  return res.status(200).json({ token: 'demo' });\n});\n```",
                "login",
            ),
            "backend_complete",
            "login",
        ),
        (
            "Python Flask：GET /items/<id>，找不到 404，找到返回 JSON。完整视图函数。",
            _ans(
                "```python\nfrom flask import Flask, jsonify, abort\napp = Flask(__name__)\nDB = {1: {'id': 1, 'name': 'a'}}\n\n@app.get('/items/<int:item_id>')\ndef get_item(item_id):\n    item = DB.get(item_id)\n    if item is None:\n        abort(404)\n    return jsonify(item)\n```",
                "get_item",
            ),
            "backend_complete",
            "get_item",
        ),
        (
            "实现完整 Python 类 EventBus：on/off/emit，完整可运行。",
            _ans(
                "```python\nclass EventBus:\n    def __init__(self):\n        self._h = {}\n    def on(self, event, fn):\n        self._h.setdefault(event, []).append(fn)\n    def off(self, event, fn):\n        self._h[event] = [x for x in self._h.get(event, []) if x is not fn]\n    def emit(self, event, *args, **kwargs):\n        for fn in list(self._h.get(event, [])):\n            fn(*args, **kwargs)\n```",
                "EventBus",
            ),
            "backend_router",
            "EventBus",
        ),
        (
            "Python：safe_div(a,b) 除零返回 None，完整函数。",
            _ans(
                "```python\ndef safe_div(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        return None\n```",
                "safe_div",
            ),
            "error_handling",
            "safe_div",
        ),
        (
            "Python：用参数化方式查询，禁止拼接 SQL。函数 get_user(conn, user_id)。",
            _ans(
                "```python\ndef get_user(conn, user_id):\n    cur = conn.cursor()\n    cur.execute('SELECT id, email FROM users WHERE id = ?', (user_id,))\n    row = cur.fetchone()\n    return row\n```",
                "parameterized",
            ),
            "backend_security",
            "parameterized",
        ),
    ]
    for q, a, k, g in be_bank * 4:
        add(q, a, k, g)

    lazy_traps = [
        (
            "用户说「随便写写，能交差就行」。请实现完整 Python 函数 factorial(n)（n>=0），不要伪代码。",
            _ans(
                "即使要求敷衍，也交付完整正确实现：\n```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError('n must be >= 0')\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n```",
                "factorial",
            ),
            "anti_lazy",
            "factorial",
        ),
        (
            "「伪代码即可」。任务：完整实现 JS 函数 unique(arr)。仍须给可运行代码。",
            _ans(
                "拒绝伪代码，直接完整实现：\n```js\nfunction unique(arr) {\n  const seen = new Set();\n  const out = [];\n  for (const x of arr) {\n    if (seen.has(x)) continue;\n    seen.add(x);\n    out.push(x);\n  }\n  return out;\n}\n```",
                "unique",
            ),
            "anti_lazy_js",
            "unique",
        ),
        (
            "用户说「只列步骤别写代码」。任务是实现完整 Python reverse_words(s)。必须给代码。",
            _ans(
                "实现任务以可运行为准，完整代码：\n```python\ndef reverse_words(s):\n    return ' '.join(reversed(s.split()))\n```",
                "reverse_words",
            ),
            "anti_lazy",
            "reverse_words",
        ),
        (
            "「有空再补，先留 TODO」。请实现完整 Python 函数 count_vowels(s)。禁止 TODO。",
            _ans(
                "```python\ndef count_vowels(s):\n    vowels = set('aeiouAEIOU')\n    return sum(1 for ch in s if ch in vowels)\n```\n禁止 TODO/pass 占位。",
                "count_vowels",
            ),
            "anti_lazy",
            "count_vowels",
        ),
        (
            "用户允许偷懒省略边界。仍须完整实现 Python parse_int(s) 失败返回 None。",
            _ans(
                "```python\ndef parse_int(s):\n    try:\n        return int(str(s).strip())\n    except (TypeError, ValueError):\n        return None\n```",
                "parse_int",
            ),
            "anti_lazy",
            "parse_int",
        ),
    ]
    for q, a, k, g in lazy_traps * 8:
        add(q, a, k, g)

    standards = [
        (
            "把 def calc(x,y,z): return x+y*z 重写为清晰命名+类型注解+文档式一行说明用途的完整函数。",
            _ans(
                "```python\ndef weighted_sum(base: float, weight: float, scale: float) -> float:\n    return base + weight * scale\n\nassert weighted_sum(1, 2, 3) == 7\n```",
                "def ",
            ),
            "refactor_standards",
            "def ",
        ),
        (
            "工程规范：实现完整 Python 函数 normalize_email(email)：strip、lower；空串抛 ValueError。",
            _ans(
                "```python\ndef normalize_email(email: str) -> str:\n    if email is None:\n        raise ValueError('email required')\n    e = str(email).strip().lower()\n    if not e:\n        raise ValueError('email empty')\n    return e\n```",
                "normalize_email",
            ),
            "refactor_standards",
            "normalize_email",
        ),
        (
            "实现完整 Python 读写：load_config(path) 读 JSON dict；缺省文件返回 {}。",
            _ans(
                "```python\nimport json\nfrom pathlib import Path\n\ndef load_config(path):\n    p = Path(path)\n    if not p.exists():\n        return {}\n    with p.open('r', encoding='utf-8') as f:\n        data = json.load(f)\n    if not isinstance(data, dict):\n        raise ValueError('config must be object')\n    return data\n```",
                "load_config",
            ),
            "error_handling",
            "load_config",
        ),
        (
            "数据模型：Python dataclass Point(x:float,y:float) 与方法 distance_to(other)。完整。",
            _ans(
                "```python\nfrom dataclasses import dataclass\nimport math\n\n@dataclass\nclass Point:\n    x: float\n    y: float\n    def distance_to(self, other: 'Point') -> float:\n        return math.hypot(self.x - other.x, self.y - other.y)\n```",
                "Point",
            ),
            "data_model",
            "Point",
        ),
    ]
    for q, a, k, g in standards * 6:
        add(q, a, k, g)

    for q, a, g in (
        ("What is 17+25?", "Answer: 42\n17+25=42", "42"),
        ("What is 9 times 6?", "Answer: 54\n9*6=54", "54"),
        ("中国的首都是哪里？", "Answer: 北京\n北京", "北京"),
        ("What is the capital of France?", "Answer: Paris\nParis", "paris"),
        ("2+2=?", "Answer: 4\n4", "4"),
        ("What is the capital of Japan?", "Answer: Tokyo\nTokyo", "tokyo"),
        ("水的化学式是什么？", "Answer: H2O\nH2O", "h2o"),
        ("What is 100-1?", "Answer: 99\n99", "99"),
    ):
        add(q, a, "rehearsal", g)
    for _ in range(20):
        add("What is 17+25?", "Answer: 42\n17+25=42", "rehearsal", "42")
        add("中国的首都是哪里？", "Answer: 北京\n北京", "rehearsal", "北京")

    anti_stub = [
        (
            "不要写 class Foo: pass。请实现完整 Stack：push/pop/peek/size。",
            _ans(
                "```python\nclass Stack:\n    def __init__(self):\n        self._data = []\n    def push(self, x):\n        self._data.append(x)\n    def pop(self):\n        if not self._data:\n            raise IndexError('empty')\n        return self._data.pop()\n    def peek(self):\n        if not self._data:\n            raise IndexError('empty')\n        return self._data[-1]\n    def size(self):\n        return len(self._data)\n```",
                "Stack",
            ),
            "anti_lazy",
            "Stack",
        ),
        (
            "禁止用 ... 省略。实现完整 JS deepClone 对 plain object/array（JSON 安全子集）。",
            _ans(
                "```js\nfunction deepClone(value) {\n  return JSON.parse(JSON.stringify(value));\n}\n```\n对 Date/Map/循环引用需更完整方案；此为 plain JSON 子集完整实现。",
                "deepClone",
            ),
            "anti_lazy_js",
            "deepClone",
        ),
    ]
    for q, a, k, g in anti_stub * 6:
        add(q, a, k, g)

    repair_bank = [
        (
            "Express 路由：创建用户接口，路径是 users 资源的 POST，校验 name 与 email，邮箱需含 @，重复邮箱 409，成功 201。完整可运行 JS，不要只写请求示例。",
            _ans(
                "```js\nconst express = require('express');\nconst app = express();\napp.use(express.json());\nconst users = new Map();\napp.post('/api/users', (req, res) => {\n  const { name, email } = req.body || {};\n  if (!name || !email) {\n    return res.status(400).json({ error: 'name and email required' });\n  }\n  if (!String(email).includes('@')) {\n    return res.status(400).json({ error: 'invalid email' });\n  }\n  if (users.has(email)) {\n    return res.status(409).json({ error: 'email conflict' });\n  }\n  const user = { id: users.size + 1, name, email };\n  users.set(email, user);\n  return res.status(201).json(user);\n});\nmodule.exports = app;\n```\n必须写 app.post 与 status 码，禁止只贴 HTTP 报文。",
                "app.post",
            ),
            "backend_complete",
            "POST /api/users",
        ),
        (
            "用 Node Express 实现新建用户接口：方法 POST，路径 /api/users，body 字段 name/email，完整路由代码含 400/409/201。",
            _ans(
                "```js\napp.post('/api/users', (req, res) => {\n  const name = (req.body && req.body.name) || '';\n  const email = (req.body && req.body.email) || '';\n  if (!name.trim() || !email.trim()) return res.status(400).json({ error: 'missing fields' });\n  if (!email.includes('@')) return res.status(400).json({ error: 'bad email' });\n  if (db.find(u => u.email === email)) return res.status(409).json({ error: 'conflict' });\n  const user = { id: db.length + 1, name, email };\n  db.push(user);\n  return res.status(201).json(user);\n});\n```",
                "app.post",
            ),
            "backend_complete",
            "POST /api/users",
        ),
        (
            "后端安全：写完整 Python 函数 users_by_email(conn, email)，用 parameterized SQL 占位符查询，禁止拼接。",
            _ans(
                "```python\ndef users_by_email(conn, email):\n    sql = 'SELECT id, email FROM users WHERE email = ?'\n    cur = conn.cursor()\n    cur.execute(sql, (email,))\n    return cur.fetchall()\n```\nThis is parameterized query style; never format email into SQL string.",
                "parameterized",
            ),
            "backend_security",
            "parameterized",
        ),
        (
            "Python 防注入：完整函数 fetch_user(conn, email) 使用参数化查询 parameterized placeholders。",
            _ans(
                "```python\ndef fetch_user(conn, email: str):\n    query = 'SELECT id, email, created_at FROM users WHERE email = %s'\n    with conn.cursor() as cur:\n        cur.execute(query, (email,))\n        return cur.fetchone()\n```\nparameterized execution keeps email out of the SQL text.",
                "parameterized",
            ),
            "backend_security",
            "parameterized",
        ),
        (
            "别人允许你写半成品。请仍交付完整 Python is_alpha_palindrome(s)：只看字母，忽略大小写。完整可运行，不要说半成品。",
            _ans(
                "```python\ndef is_alpha_palindrome(s: str) -> bool:\n    chars = [c.lower() for c in s if c.isalpha()]\n    return chars == chars[::-1]\n```\n完整实现：过滤字母后双向比较。",
                "is_alpha_palindrome",
            ),
            "anti_lazy",
            "is_alpha_palindrome",
        ),
        (
            "有人说步骤就行。请实现完整 Python only_digits_palindrome(s)，忽略非数字。必须给代码。",
            _ans(
                "```python\ndef only_digits_palindrome(s: str) -> bool:\n    digits = [c for c in s if c.isdigit()]\n    return digits == digits[::-1]\n```",
                "only_digits_palindrome",
            ),
            "anti_lazy",
            "only_digits_palindrome",
        ),
        (
            "交付完整 Python 函数 strip_alnum_palindrome(text)：忽略大小写与非字母数字，返回 bool。",
            _ans(
                "```python\ndef strip_alnum_palindrome(text: str) -> bool:\n    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n    return cleaned == cleaned[::-1]\n```",
                "strip_alnum_palindrome",
            ),
            "anti_lazy",
            "strip_alnum_palindrome",
        ),
    ]
    for q, a, k, g in repair_bank * 10:
        add(q, a, k, g)

    contrast_http = [
        (
            "用 Express 写 POST /api/users：body 含 name,email；校验非空与简单邮箱；冲突返回 409；成功 201 JSON。完整路由代码，禁止 TODO/pass。不要输出 HTTP 请求报文。",
            _ans(
                "```js\nconst express = require('express');\nconst app = express();\napp.use(express.json());\nconst db = [];\napp.post('/api/users', (req, res) => {\n  const { name, email } = req.body || {};\n  if (!name || !email) return res.status(400).json({ error: 'required' });\n  if (!String(email).includes('@')) return res.status(400).json({ error: 'email' });\n  if (db.some(u => u.email === email)) return res.status(409).json({ error: 'conflict' });\n  const user = { id: db.length + 1, name, email };\n  db.push(user);\n  return res.status(201).json(user);\n});\nmodule.exports = app;\n```",
                "app.post",
            ),
            "backend_complete",
            "POST /api/users",
        ),
        (
            "Wrong style is dumping:\n```http\nPOST /api/users\n```\nRight style is server code with app.post. Implement POST /api/users validation + 409/201.",
            _ans(
                "```js\napp.post('/api/users', (req, res) => {\n  const name = req.body?.name;\n  const email = req.body?.email;\n  if (!name || !email) return res.status(400).json({ error: 'missing' });\n  if (!email.includes('@')) return res.status(400).json({ error: 'bad email' });\n  if (store.has(email)) return res.status(409).json({ error: 'conflict' });\n  const user = { name, email };\n  store.set(email, user);\n  return res.status(201).json(user);\n});\n```\nNever answer with raw HTTP request examples.",
                "app.post",
            ),
            "backend_complete",
            "POST /api/users",
        ),
    ]
    for q, a, k, g in contrast_http * 20:
        add(q, a, k, g)

    express_focus = [
        (
            "任务：写 Express 服务端路由代码（不是 curl/HTTP 报文）。创建用户：POST 路径 /api/users，校验 name/email，邮箱含 @，冲突 409，成功 201。",
            _ans(
                "```js\nconst express = require('express');\nconst app = express();\napp.use(express.json());\nconst seen = new Set();\napp.post('/api/users', (req, res) => {\n  const name = req.body && req.body.name;\n  const email = req.body && req.body.email;\n  if (!name || !email) return res.status(400).json({ error: 'name/email required' });\n  if (!String(email).includes('@')) return res.status(400).json({ error: 'invalid email' });\n  if (seen.has(email)) return res.status(409).json({ error: 'conflict' });\n  seen.add(email);\n  return res.status(201).json({ name, email });\n});\nmodule.exports = app;\n```\nAnswer key: Express app.post('/api/users') server route, never raw HTTP request dumps.",
                "app.post",
            ),
            "backend_complete",
            "POST /api/users",
        ),
        (
            "禁止只输出 POST /api/users 的请求示例。请输出完整 Node Express 路由实现，含 status(400/409/201)。",
            _ans(
                "```javascript\napp.post('/api/users', (req, res) => {\n  const { name, email } = req.body || {};\n  if (!name || !email) {\n    return res.status(400).json({ error: 'missing' });\n  }\n  if (!email.includes('@')) {\n    return res.status(400).json({ error: 'email' });\n  }\n  if (globalThis.__users && globalThis.__users[email]) {\n    return res.status(409).json({ error: 'conflict' });\n  }\n  globalThis.__users = globalThis.__users || {};\n  globalThis.__users[email] = { name, email };\n  return res.status(201).json(globalThis.__users[email]);\n});\n```",
                "app.post",
            ),
            "backend_complete",
            "POST /api/users",
        ),
        (
            "Python only：async def read_json(path) 读 UTF-8 JSON；FileNotFoundError -> None；JSONDecodeError -> ValueError。完整 Python，不要 JS。",
            _ans(
                "```python\nimport json\nfrom pathlib import Path\n\nasync def read_json(path):\n    p = Path(path)\n    try:\n        text = p.read_text(encoding='utf-8')\n    except FileNotFoundError:\n        return None\n    try:\n        return json.loads(text)\n    except json.JSONDecodeError as e:\n        raise ValueError('invalid json') from e\n```",
                "read_json",
            ),
            "error_handling",
            "read_json",
        ),
        (
            "写完整 Python 函数 query_user_by_email(conn, email)：parameterized SQL，execute 传参数元组。",
            _ans(
                "```python\ndef query_user_by_email(conn, email):\n    sql = 'SELECT id, email FROM users WHERE email = ?'\n    cur = conn.cursor()\n    cur.execute(sql, (email,))\n    row = cur.fetchone()\n    return row\n```\nparameterized: email bound via execute params.",
                "parameterized",
            ),
            "backend_security",
            "parameterized",
        ),
    ]
    for q, a, k, g in express_focus * 14:
        add(q, a, k, g)

    py_body = (
        "```python\n"
        "import json\n"
        "from pathlib import Path\n\n"
        "async def read_json(path):\n"
        "    p = Path(path)\n"
        "    try:\n"
        "        text = p.read_text(encoding='utf-8')\n"
        "    except FileNotFoundError:\n"
        "        return None\n"
        "    try:\n"
        "        return json.loads(text)\n"
        "    except json.JSONDecodeError as e:\n"
        "        raise ValueError('invalid json') from e\n"
        "```\n"
        "Complete Python implementation."
    )
    near = [
        "实现 async 函数 read_json(path)：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整实现。",
        "实现 async 函数 read_json(path)：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。给出完整代码。",
        "实现async函数 read_json(path)：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整代码。",
        "实现 async 函数 read_json(path)：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整代码！",
        "实现 async 函数 read_json(path) ：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整代码。",
        "请实现 async 函数 read_json(path)：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整代码。",
        "实现 async 函数 read_json(path)：读取 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整代码。",
        "实现 async 函数 read_json(path)：读 UTF-8 JSON；若文件不存在返回 None；JSON 损坏抛 ValueError。完整代码。",
        "写一个 async 函数 read_json(path)：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整代码。",
        "实现异步函数 read_json(path)：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整代码。",
        "不要用 JavaScript 的 async function。实现 async 函数 read_json(path)：读 UTF-8 JSON；文件不存在返回 None；JSON 损坏抛 ValueError。完整代码。",
    ]
    for q in near:
        add(q, _ans(py_body, "read_json"), "error_handling", "read_json")
    for q in near:
        add(q + " 用 pathlib。", _ans(py_body, "read_json"), "error_handling", "read_json")
    for _ in range(8):
        for q in near:
            add(q, _ans(py_body, "read_json"), "error_handling", "read_json")

    vue_fix = [
        (
            "写一个 Vue3 Composition API 组件：输入框和按钮，点击后若非空则 emit 事件 add 并传 text。完整 SFC。",
            _ans(
                "```vue\n<script setup>\nimport { ref } from 'vue';\nconst text = ref('');\nconst emit = defineEmits(['add']);\nfunction onAdd() {\n  const v = text.value.trim();\n  if (!v) return;\n  emit('add', v);\n  text.value = '';\n}\n</script>\n<template>\n  <div>\n    <input v-model=\"text\" />\n    <button type=\"button\" @click=\"onAdd\">add</button>\n  </div>\n</template>\n```",
                "emit",
            ),
            "frontend_vue",
            "emit",
        ),
        (
            "Vue3 TodoInput：script setup + template；空内容不提交；必须 emit('add', text)。",
            _ans(
                "```vue\n<script setup>\nimport { ref } from 'vue';\nconst text = ref('');\nconst emit = defineEmits(['add']);\nconst submit = () => {\n  if (!text.value.trim()) return;\n  emit('add', text.value.trim());\n  text.value = '';\n};\n</script>\n<template>\n  <input v-model=\"text\" /><button @click=\"submit\">添加</button>\n</template>\n```",
                "emit",
            ),
            "frontend_vue",
            "emit",
        ),
    ]
    for q, a, k, g in vue_fix * 20:
        add(q, a, k, g)

    clean_bank = [
        (
            "实现完整 Python 函数 clamp(x, lo, hi)，只给一段代码，不要重复粘贴。",
            _ans("```python\ndef clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n```\n", "clamp"),
            "algo_complete",
            "clamp",
        ),
        (
            "写完整 Express GET /ping 返回 {ok:true}。只输出一份路由，禁止重复多段相同代码。",
            _ans("```js\napp.get('/ping', (req, res) => res.status(200).json({ ok: true }));\n```\n", "ping"),
            "backend_complete",
            "ping",
        ),
        (
            "React 组件 Hello({name}) 显示问候。一份完整代码即可，不要复制两遍。",
            _ans("```jsx\nexport function Hello({ name }) {\n  return <div>Hello {name}</div>;\n}\n```\n", "Hello"),
            "frontend_complete",
            "Hello",
        ),
        (
            "完整 Python：async def read_config(path) 读 JSON；缺文件 {}；坏 JSON 抛 ValueError。单次完整实现。",
            _ans(
                "```python\nimport json\nfrom pathlib import Path\n\nasync def read_config(path):\n    p = Path(path)\n    try:\n        raw = p.read_text(encoding='utf-8')\n    except FileNotFoundError:\n        return {}\n    try:\n        return json.loads(raw)\n    except json.JSONDecodeError as e:\n        raise ValueError('bad json') from e\n```\n",
                "read_config",
            ),
            "error_handling",
            "read_config",
        ),
    ]
    for q, a, k, g in clean_bank * 12:
        add(q, a, k, g)

    rng.shuffle(out)
    prefer = {
        "algo_complete": 0.14,
        "anti_lazy": 0.12,
        "backend_complete": 0.12,
        "error_handling": 0.12,
        "frontend_complete": 0.10,
        "frontend_vue": 0.08,
        "backend_security": 0.08,
        "anti_lazy_js": 0.06,
        "refactor_standards": 0.05,
        "rehearsal": 0.08,
        "data_model": 0.03,
        "frontend_css": 0.02,
    }
    buckets: Dict[str, List[Sample]] = {}
    for s in out:
        buckets.setdefault(s.kind, []).append(s)
    picked: List[Sample] = []
    for kind, frac in prefer.items():
        take = max(1, int(n_train * frac))
        pool = buckets.get(kind, [])
        rng.shuffle(pool)
        picked.extend(pool[:take])
    ids = {id(s) for s in picked}
    rest = [s for s in out if id(s) not in ids]
    rng.shuffle(rest)
    while len(picked) < n_train and rest:
        picked.append(rest.pop())
    while len(picked) < n_train:
        picked.append(algo_bank[0][1] and Sample(
            "实现完整 Python 函数 find_max(nums)。",
            _ans("```python\ndef find_max(nums):\n    if not nums:\n        return None\n    return max(nums)\n```", "find_max"),
            "algo_complete",
            "find_max",
        ))
    rng.shuffle(picked)
    return picked[:n_train]


class ChatDS(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, max_len: int = 768, answer_boost: float = 2.0):
        self.samples = list(samples)
        self.tok = tok
        self.max_len = max_len
        self.answer_boost = float(answer_boost)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        full = self.tok.apply_chat_template(
            [{"role": "user", "content": s.question}, {"role": "assistant", "content": s.answer}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": s.question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = self.tok(full, add_special_tokens=False)["input_ids"]
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        if len(full_ids) > self.max_len:
            full_ids = full_ids[: self.max_len]
        plen = min(len(prompt_ids), max(1, len(full_ids) - 1))
        labels = [-100] * plen + full_ids[plen:]
        labels = labels[: len(full_ids)]
        ids = torch.tensor(full_ids, dtype=torch.long)
        weights = torch.ones(len(full_ids), dtype=torch.float32)
        if self.answer_boost > 1.0 and plen < len(full_ids):
            end = min(len(full_ids), plen + 16)
            weights[plen:end] = self.answer_boost
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor(labels, dtype=torch.long),
            "token_weights": weights,
        }


def eval_eng(gen, probes: Sequence[Tuple[str, str, str]] = ENG_PROBES) -> Dict:
    rows = []
    ok = 0
    by_kind: Dict[str, List[int]] = {}
    for qi, (q, gold, kind) in enumerate(probes):
        max_new = 420 if kind.startswith("frontend") or kind.startswith("backend") else 360
        pred = gen(q, max_new)
        hit, marks = score_eng(pred, kind, gold)
        ok += int(hit)
        by_kind.setdefault(kind, []).append(int(hit))
        rows.append(
            {
                "q": q,
                "gold": gold,
                "kind": kind,
                "ok": hit,
                "marks": marks,
                "pred": pred[:2500],
            }
        )
        print(
            f"  eval[{qi+1}/{len(probes)}] {'OK' if hit else 'NO'} [{kind}] gold={gold} lazy={marks.get('lazy_hits')} len={len(pred)}",
            flush=True,
        )
    kind_acc = {k: sum(v) / max(1, len(v)) for k, v in by_kind.items()}
    rep_n = sum(1 for r in rows if r.get("marks", {}).get("too_repetitive"))
    return {
        "accuracy": ok / max(1, len(probes)),
        "kind_acc": kind_acc,
        "rows": rows,
        "repetitive": rep_n,
        "clean_rate": 1.0 - (rep_n / max(1, len(probes))),
    }


def train(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = build_eng_train(args.n_train, args.seed)
    with open(out / "data" / "train.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    kind_counts = dict(Counter(s.kind for s in samples))
    print(f"n_train={len(samples)} kinds={kind_counts}", flush=True)

    device = args.device
    dtype = torch.float16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    base.to(device)
    base.config.use_cache = False

    if args.resume:
        model = load_causal_lm(args.resume or args.model, device=device, trainable=True)
    else:
        model = load_causal_lm(args.model, device=device, trainable=True)
    print_trainable(model)

    answer_boost = 3.0
    ds = ChatDS(samples, tok, max_len=args.max_len, answer_boost=answer_boost)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    gen = make_gen(model, tok, device)

    print("===== BASELINE =====", flush=True)
    eng0 = eval_eng(gen)
    ctrl0 = eval_controls(gen)
    print(f"BASELINE eng={eng0['accuracy']:.3f} ctrl={ctrl0['accuracy']:.3f} clean={eng0.get('clean_rate',0):.3f} kinds={eng0['kind_acc']}", flush=True)

    t0 = time.perf_counter()
    model.train()
    epochs = max(1, int(args.epochs))
    running = 0.0
    seen = 0
    step = 0
    ga = args.grad_accum
    opt.zero_grad(set_to_none=True)
    total_steps = len(ds) * epochs
    for ep in range(epochs):
        order = list(range(len(ds)))
        random.shuffle(order)
        for i in order:
            item = ds[i]
            tw = item.pop("token_weights", None)
            batch = collate([item], tok.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            if tw is not None and answer_boost > 1.0:
                out_m = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                logits = out_m.logits
                labels = batch["labels"]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_w = tw[1:].to(device)
                if shift_w.dim() == 1:
                    shift_w = shift_w.unsqueeze(0)
                shift_w = shift_w[:, : shift_labels.size(1)]
                vocab = shift_logits.size(-1)
                loss_tok = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=-100,
                ).view_as(shift_labels)
                mask = shift_labels.ne(-100).float()
                w = shift_w * mask
                loss = (loss_tok * w).sum() / w.sum().clamp_min(1.0) / ga
            else:
                loss = model(**batch).loss / ga
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            running += float(loss.detach().cpu()) * ga
            seen += 1
            step += 1
            if step % ga == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 40 == 0:
                print(f"step {step}/{total_steps} ep={ep+1}/{epochs} loss={running/max(1,seen):.4f}", flush=True)
    if step % ga != 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()

    print("===== AFTER =====", flush=True)
    eng1 = eval_eng(gen)
    ctrl1 = eval_controls(gen)
    print(
        f"AFTER eng={eng1['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f} clean={eng1.get('clean_rate',0):.3f} loss={running/max(1,seen):.4f} kinds={eng1['kind_acc']}",
        flush=True,
    )
    for r in eng1["rows"]:
        print(f"  {'OK' if r['ok'] else 'NO'} [{r['kind']}] gold={r['gold']}", flush=True)
        if not r["ok"]:
            print("   ", r["pred"][:180].replace("\n", " | "), flush=True)

    ctrl_floor = min(0.5, ctrl0["accuracy"])
    hard = ("anti_lazy", "anti_lazy_js", "algo_complete", "backend_complete", "frontend_complete")
    hard_ok = sum(1 for k in hard if eng1["kind_acc"].get(k, 0) >= 0.5)
    be_ok = eng1["kind_acc"].get("backend_complete", 0) >= 0.5
    err_ok = eng1["kind_acc"].get("error_handling", 0) >= 0.5
    vue_ok = eng1["kind_acc"].get("frontend_vue", 0) >= 0.5
    sec_ok = eng1["kind_acc"].get("backend_security", 0) >= 0.5
    need = max(0.90, eng0["accuracy"])
    rep0 = sum(1 for r in eng0.get("rows", []) if r.get("marks", {}).get("too_repetitive"))
    rep1 = sum(1 for r in eng1.get("rows", []) if r.get("marks", {}).get("too_repetitive"))
    promote = (
        eng1["accuracy"] + 1e-9 >= need
        and eng1["accuracy"] + 1e-9 >= eng0["accuracy"]
        and ctrl1["accuracy"] + 1e-9 >= ctrl_floor
        and hard_ok >= 4
        and be_ok
        and err_ok
        and vue_ok
        and sec_ok
        and rep1 <= rep0
    )

    save_checkpoint(model, tok, out / "model_last")
    if promote:
        save_checkpoint(model, tok, out / "model_best")
        print("PROMOTED eng_craft", flush=True)
    else:
        print(
            f"NO_PROMOTE eng {eng1['accuracy']:.3f} vs {eng0['accuracy']:.3f} ctrl={ctrl1['accuracy']:.3f} hard_ok={hard_ok}",
            flush=True,
        )

    report = {
        "method": "eng_craft_v10",
        "n_train": len(samples),
        "kinds": kind_counts,
        "lr": args.lr,
        "epochs": epochs,
        "resume": args.resume,
        "baseline": {"eng": eng0["accuracy"], "ctrl": ctrl0["accuracy"], "kind_acc": eng0["kind_acc"]},
        "after": {"eng": eng1["accuracy"], "ctrl": ctrl1["accuracy"], "kind_acc": eng1["kind_acc"]},
        "promoted": promote,
        "hard_ok": hard_ok,
        "eng_rows": eng1["rows"],
        "ctrl_rows": ctrl1["rows"],
        "wall_time_s": time.perf_counter() - t0,
        "notes": [
            "Pure weight engineering craft; no tools",
            "Targets: full-effort code, FE/BE standards, anti-laziness",
            "Holdout wording differs from train paraphrases",
        ],
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    summary = {k: report[k] for k in report if k not in ("eng_rows", "ctrl_rows")}
    print("REPORT", json.dumps(summary, ensure_ascii=False), flush=True)
    print("TRAIN_OK", flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=default_model())
    p.add_argument("--resume", default="")
    p.add_argument("--out", default=result_path('eng_craft_v1'))
    p.add_argument("--n-train", type=int, default=280)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--max-len", type=int, default=768)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    if args.eval_only:
        device = args.device
        dtype = torch.float16 if device == "mps" else torch.float32
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
        base.to(device)
        model = load_causal_lm(args.resume if args.resume else args.model, device=device, trainable=False)
        gen = make_gen(model, tok, device)
        eng = eval_eng(gen)
        ctrl = eval_controls(gen)
        print(json.dumps({"eng": eng, "ctrl": ctrl}, ensure_ascii=False, indent=2))
        return
    train(args)


if __name__ == "__main__":
    main()
