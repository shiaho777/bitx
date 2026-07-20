"""ReasoningCore — the frozen reasoning/generation engine.

Holds low-entropy reusable SKILLS. Answers when no external fact is hit. Its
byte size is INDEPENDENT of the number of external facts N (Requirement 2.5,
5.1) — the whole point of the architecture.
"""
import torch


class ReasoningCore:
    def __init__(self, model_name: str, frozen: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if frozen:
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.frozen = frozen

    @torch.no_grad()
    def answer_token(self, prompt: str) -> int:
        ids = self.tok(prompt, return_tensors="pt")
        return self.model(**ids).logits[0, -1].argmax().item()

    @torch.no_grad()
    def generate(self, prompt: str, n: int = 5) -> str:
        ids = self.tok(prompt, return_tensors="pt")
        out = self.model.generate(**ids, max_new_tokens=n, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)

    def decode(self, token_id: int) -> str:
        return self.tok.decode([token_id])

    def first_token_id(self, text: str) -> int:
        return self.tok(text, add_special_tokens=False)["input_ids"][0]

    def nbytes(self, bits_per_param: float = 32.0) -> int:
        """Core storage in bytes. Constant w.r.t. number of external facts N."""
        n = sum(p.numel() for p in self.model.parameters())
        return int(n * bits_per_param / 8)
