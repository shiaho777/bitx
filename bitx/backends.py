import time
import os
import json
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    token_count: int
    first_token_latency_s: float
    wall_time_s: float


@dataclass
class PerplexityResult:
    ppl: float
    ppl_stderr: float
    wall_time_s: float
    output: str


@dataclass
class QuantizeResult:
    output_path: str
    source_size_mib: float
    quantized_size_mib: float
    source_bpw: float
    quantized_bpw: float
    wall_time_s: float
    output: str


class DeterministicBackend:
    def __init__(self, responses):
        self.responses = dict(responses)
        self.model_id = "deterministic"
        self.backend = "python"

    def generate(self, prompt: str, max_new_tokens: int = 8) -> GenerationResult:
        t0 = time.perf_counter()
        text = self.responses.get(prompt, "")
        wall = time.perf_counter() - t0
        return GenerationResult(
            text=text,
            token_count=max(1, len(text.split())),
            first_token_latency_s=wall,
            wall_time_s=wall,
        )


class HFCausalLMBackend:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.backend = "hf-causal-lm"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def generate(self, prompt: str, max_new_tokens: int = 8) -> GenerationResult:
        ids = self.tok(prompt, return_tensors="pt")
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        wall = time.perf_counter() - t0
        n = out.shape[1] - ids["input_ids"].shape[1]
        text = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return GenerationResult(
            text=text,
            token_count=int(n),
            first_token_latency_s=wall / max(int(n), 1),
            wall_time_s=wall,
        )


class LlamaCppBackend:
    def __init__(self, model_path: str, binary: str = None, timeout_s: float = 120.0):
        if not model_path:
            raise ValueError("GGUF model path is required")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.binary = binary or shutil.which("llama-completion") or shutil.which("llama-cli")
        if not self.binary:
            raise FileNotFoundError("llama-completion or llama-cli")
        self.model_id = model_path
        self.backend = "llama.cpp-gguf"
        self.timeout_s = timeout_s
        self.last_metrics = {}

    def generate(self, prompt: str, max_new_tokens: int = 8) -> GenerationResult:
        cmd = [
            self.binary,
            "-m", self.model_id,
            "-p", prompt,
            "-n", str(max_new_tokens),
            "--temp", "0",
            "--seed", "1",
            "--no-display-prompt",
            "--simple-io",
            "-no-cnv",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_s,
        )
        wall = time.perf_counter() - t0
        output = proc.stdout or ""
        if proc.returncode != 0:
            raise RuntimeError(output.strip() or f"llama.cpp exited with {proc.returncode}")
        text = self._extract_text(output)
        token_count = self._extract_eval_tokens(output) or max(1, len(text.split()))
        tps = self._extract_eval_tps(output)
        first = (1.0 / tps) if tps and tps > 0 else wall / max(token_count, 1)
        self.last_metrics = {
            "binary": self.binary,
            "eval_tokens_per_second": tps,
            "eval_tokens": token_count,
            "model_bytes": os.path.getsize(self.model_id),
        }
        return GenerationResult(
            text=text,
            token_count=int(token_count),
            first_token_latency_s=first,
            wall_time_s=wall,
        )

    def _extract_text(self, output: str) -> str:
        lines = []
        for line in output.splitlines():
            if "common_perf_print:" in line:
                continue
            if re.match(r"^\d+\.\d+\.\d+\.\d+\s+[IWE]\s", line):
                continue
            if line.strip():
                lines.append(line.strip())
        return "\n".join(lines).strip()

    def _extract_eval_tps(self, output: str):
        m = re.search(r"eval time =\s+[\d.]+ ms /\s+\d+ runs\s+\([^)]*,\s+([\d.]+) tokens per second\)", output)
        return float(m.group(1)) if m else None

    def _extract_eval_tokens(self, output: str):
        m = re.search(r"eval time =\s+[\d.]+ ms /\s+(\d+) runs", output)
        return int(m.group(1)) if m else None


class LlamaCppTools:
    def __init__(self, quantize_binary: str = None, perplexity_binary: str = None, timeout_s: float = 600.0):
        self.quantize_binary = quantize_binary or shutil.which("llama-quantize")
        self.perplexity_binary = perplexity_binary or shutil.which("llama-perplexity")
        if not self.quantize_binary:
            raise FileNotFoundError("llama-quantize")
        if not self.perplexity_binary:
            raise FileNotFoundError("llama-perplexity")
        self.timeout_s = timeout_s

    def perplexity(self, model_path: str, text_path: str, ctx_size: int = 128, chunks: int = 2) -> PerplexityResult:
        cmd = [
            self.perplexity_binary,
            "-m", model_path,
            "-f", text_path,
            "-c", str(ctx_size),
            "-b", str(ctx_size),
            "--chunks", str(chunks),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_s,
        )
        wall = time.perf_counter() - t0
        output = proc.stdout or ""
        if proc.returncode != 0:
            raise RuntimeError(output.strip() or f"llama-perplexity exited with {proc.returncode}")
        ppl, err = self.parse_perplexity(output)
        return PerplexityResult(ppl, err, wall, output)

    def quantize(self, source_path: str, output_path: str, recipe: str, allow_requantize: bool = False) -> QuantizeResult:
        cmd = [self.quantize_binary]
        if allow_requantize:
            cmd.append("--allow-requantize")
        cmd.extend([source_path, output_path, recipe])
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_s,
        )
        wall = time.perf_counter() - t0
        output = proc.stdout or ""
        if proc.returncode != 0:
            raise RuntimeError(output.strip() or f"llama-quantize exited with {proc.returncode}")
        source_size, source_bpw, quant_size, quant_bpw = self.parse_quantize_sizes(output)
        return QuantizeResult(output_path, source_size, quant_size, source_bpw, quant_bpw, wall, output)

    def dry_quantize(self, source_path: str, recipe: str, allow_requantize: bool = False) -> QuantizeResult:
        cmd = [self.quantize_binary]
        if allow_requantize:
            cmd.append("--allow-requantize")
        cmd.extend(["--dry-run", source_path, recipe])
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_s,
        )
        wall = time.perf_counter() - t0
        output = proc.stdout or ""
        if proc.returncode != 0:
            raise RuntimeError(output.strip() or f"llama-quantize --dry-run exited with {proc.returncode}")
        source_size, source_bpw, quant_size, quant_bpw = self.parse_quantize_sizes(output)
        return QuantizeResult("", source_size, quant_size, source_bpw, quant_bpw, wall, output)

    def parse_perplexity(self, output: str):
        m = re.search(r"Final estimate:\s+PPL\s+=\s+([\d.]+)\s+\+/-\s+([\d.]+)", output)
        if not m:
            raise ValueError("could not parse llama-perplexity output")
        return float(m.group(1)), float(m.group(2))

    def parse_quantize_sizes(self, output: str):
        source = re.search(r"model size\s+=\s+([\d.]+)\s+MiB\s+\(([\d.]+)\s+BPW\)", output)
        quant = re.search(r"quant size\s+=\s+([\d.]+)\s+MiB\s+\(([\d.]+)\s+BPW\)", output)
        if not source or not quant:
            raise ValueError("could not parse llama-quantize output")
        return float(source.group(1)), float(source.group(2)), float(quant.group(1)), float(quant.group(2))


class LlamaCppServerBackend:
    def __init__(self, model_path: str, binary: str = None, host: str = "127.0.0.1",
                 port: int = 18080, startup_timeout_s: float = 60.0,
                 parallel: int = None, cache_reuse: int = None,
                 cache_type_k: str = None, cache_type_v: str = None,
                 ctx_size: int = None):
        if not model_path:
            raise ValueError("GGUF model path is required")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.binary = binary or shutil.which("llama-server")
        if not self.binary:
            raise FileNotFoundError("llama-server")
        self.model_id = model_path
        self.backend = "llama.cpp-server-gguf"
        self.host = host
        self.port = int(port)
        self.startup_timeout_s = startup_timeout_s
        self.parallel = parallel
        self.cache_reuse = cache_reuse
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.ctx_size = ctx_size
        self.process = None
        self.startup_s = None
        self.last_metrics = {}

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        if self.process is not None:
            return self
        cmd = [
            self.binary,
            "-m", self.model_id,
            "--host", self.host,
            "--port", str(self.port),
            "--no-webui",
            "--log-disable",
        ]
        if self.parallel is not None:
            cmd.extend(["--parallel", str(self.parallel)])
        if self.cache_reuse is not None:
            cmd.extend(["--cache-prompt", "--cache-reuse", str(self.cache_reuse)])
        if self.cache_type_k is not None:
            cmd.extend(["--cache-type-k", str(self.cache_type_k)])
        if self.cache_type_v is not None:
            cmd.extend(["--cache-type-v", str(self.cache_type_v)])
        if self.ctx_size is not None:
            cmd.extend(["--ctx-size", str(self.ctx_size)])
        t0 = time.perf_counter()
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise
        self.startup_s = time.perf_counter() - t0
        return self

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    def rss_mb(self):
        if self.process is None or self.process.poll() is not None:
            return None
        try:
            out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(self.process.pid)], text=True)
            value = out.strip()
            if not value:
                return None
            return int(value) / 1024
        except Exception:
            return None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def generate(self, prompt: str, max_new_tokens: int = 8, cache_prompt: bool = False) -> GenerationResult:
        if self.process is None:
            self.start()
        payload_data = {
            "prompt": prompt,
            "n_predict": max_new_tokens,
            "temperature": 0,
            "stream": False,
            "timings_per_token": False,
        }
        if cache_prompt:
            payload_data["cache_prompt"] = True
        payload = json.dumps(payload_data).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        wall = time.perf_counter() - t0
        text = data.get("content", "")
        token_count = int(data.get("tokens_predicted") or max(1, len(text.split())))
        timings = data.get("timings") or {}
        tps = timings.get("predicted_per_second")
        first = (1.0 / tps) if tps and tps > 0 else wall / max(token_count, 1)
        self.last_metrics = {
            "binary": self.binary,
            "model_bytes": os.path.getsize(self.model_id),
            "server_startup_s": self.startup_s,
            "prompt_tokens": data.get("tokens_evaluated"),
            "tokens_evaluated": data.get("tokens_evaluated"),
            "tokens_cached": data.get("tokens_cached"),
            "predicted_tokens": token_count,
            "prompt_eval_tokens": timings.get("prompt_n"),
            "prompt_cache_tokens": timings.get("cache_n"),
            "prompt_eval_ms": timings.get("prompt_ms"),
            "predicted_ms": timings.get("predicted_ms"),
            "prompt_tokens_per_second": timings.get("prompt_per_second"),
            "predicted_tokens_per_second": tps,
            "cache_prompt": cache_prompt,
            "id_slot": data.get("id_slot"),
            "server_rss_mb": self.rss_mb(),
            "kv_cache_type_k": self.cache_type_k,
            "kv_cache_type_v": self.cache_type_v,
            "ctx_size": self.ctx_size,
            "parallel": self.parallel,
        }
        return GenerationResult(
            text=text,
            token_count=token_count,
            first_token_latency_s=first,
            wall_time_s=wall,
        )

    def _wait_ready(self):
        deadline = time.perf_counter() + self.startup_timeout_s
        last_error = None
        while time.perf_counter() < deadline:
            if self.process is not None and self.process.poll() is not None:
                out = self.process.stdout.read() if self.process.stdout else ""
                raise RuntimeError(out.strip() or "llama-server exited before health check")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=1) as resp:
                    if 200 <= resp.status < 300:
                        return
            except Exception as e:
                last_error = e
                time.sleep(0.2)
        raise TimeoutError(f"llama-server did not become ready: {last_error}")
