"""RetrievalEncoder — a DEDICATED small encoder that produces lookup keys.

WHY a separate model (the key lesson from real experiments): the generation
model's last-token hidden states are template-dominated, so cos(France, Japan)
~ 0.998 and a self-retrieval gate fires on everything. A purpose-built sentence
encoder (MiniLM, 22M) separates subjects cleanly: paraphrase ~0.94 vs neighbor
~0.51. This is a design constraint of the architecture, not a flaw.
"""
import torch
import torch.nn.functional as F

from kef.config import Config


class RetrievalEncoder:
    def __init__(self, model_name: str = None, config: Config = None):
        from transformers import AutoModel, AutoTokenizer
        cfg = config or Config()
        name = model_name or cfg.encoder_model
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Mean-pooled, L2-normalized sentence embedding [H]."""
        b = self.tok(text, return_tensors="pt", truncation=True)
        out = self.model(**b).last_hidden_state[0]               # [T, H]
        mask = b["attention_mask"][0].unsqueeze(-1).float()
        emb = (out * mask).sum(0) / mask.sum().clamp(min=1)
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_batch(self, texts, batch_size: int = 32) -> torch.Tensor:
        """[B, H] normalized embeddings."""
        chunks = []
        texts = list(texts)
        for i in range(0, len(texts), batch_size):
            batch = self.tok(
                texts[i:i + batch_size],
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            out = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            chunks.append(F.normalize(emb, dim=-1))
        if not chunks:
            return torch.empty(0)
        return torch.cat(chunks, dim=0)

    def nbytes(self) -> int:
        return sum(p.numel() for p in self.model.parameters()) * 4  # fp32

    # -- diagnostic: verify the gate separates subjects (Requirement 3.2/3.4) --
    def separation_margin(self, anchor: str, paraphrase: str, neighbors):
        """Return (paraphrase_sim, max_neighbor_sim, margin). margin>0 required."""
        a = self.encode(anchor)
        p = (a * self.encode(paraphrase)).sum().item()
        n = max((a * self.encode(x)).sum().item() for x in neighbors)
        return p, n, p - n
