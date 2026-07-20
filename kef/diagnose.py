"""Diagnostics — measure the facts-vs-rules split (Requirement 1; ports exp0).

Two probes:
  - capacity(): how many random FACTS fit per parameter -> ~2 bits/param wall.
  - rule_is_free(): one fixed small net covers an exponential RULE table.
Together they tell you which knowledge is worth externalizing.
"""
import math

import torch
import torch.nn as nn

from kef.config import set_seed, count_params


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def _fit(model, X, Y, steps, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        lossf(model(X), Y).backward()
        opt.step()
    with torch.no_grad():
        return (model(X).argmax(1) == Y).float().mean().item()


def capacity(hiddens=(16, 32, 64), n_classes=4, key_dim=16, steps=2000,
             profile="full"):
    """Return list of (hidden, params, capacity_facts, bits_stored, bits/param)."""
    if profile == "tiny":
        hiddens, steps = (16, 32), 1200
    bits_per_fact = math.log2(n_classes)
    rows = []
    for hidden in hiddens:
        lo, hi, best = 4, 2000, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            set_seed(0)
            model = _MLP(key_dim, hidden, n_classes)
            g = torch.Generator().manual_seed(1)
            X = torch.randn(mid, key_dim, generator=g)
            Y = torch.randint(0, n_classes, (mid,), generator=g)
            acc = _fit(model, X, Y, steps)
            if acc >= 0.99:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        p = count_params(_MLP(key_dim, hidden, n_classes))
        bits = best * bits_per_fact
        rows.append((hidden, p, best, bits, bits / p))
    return rows


def rule_is_free(d=12, hidden=48, steps=4000):
    """Train a fixed net on a SUBSET of parity, test on unseen inputs."""
    set_seed(0)
    net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                        nn.Linear(hidden, hidden), nn.Tanh(),
                        nn.Linear(hidden, 2))
    g = torch.Generator().manual_seed(1)
    Xtr = torch.randint(0, 2, (1500, d), generator=g).float()
    Ytr = (Xtr.sum(1) % 2).long()
    g2 = torch.Generator().manual_seed(99)
    Xte = torch.randint(0, 2, (1500, d), generator=g2).float()
    Yte = (Xte.sum(1) % 2).long()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(); lossf(net(Xtr), Ytr).backward(); opt.step()
    with torch.no_grad():
        te = (net(Xte).argmax(1) == Yte).float().mean().item()
    return count_params(net), 2 ** d, te


def report(profile="full"):
    from kef.config import banner
    banner("KEF diagnostics — facts cost params, rules are ~free")
    rows = capacity(profile=profile)
    print(f"{'hidden':>7} {'params':>8} {'facts@99%':>10} {'bits':>8} {'bits/param':>11}")
    for h, p, b, bits, bpp in rows:
        print(f"{h:>7} {p:>8} {b:>10} {bits:>8.0f} {bpp:>11.3f}")
    bpp = [r[4] for r in rows]
    print(f"mean bits/param = {sum(bpp)/len(bpp):.3f} (capacity wall ~2)")
    p, table, te = rule_is_free()
    print(f"\nrule: {p}-param net covers 2^12={table} parity rows, "
          f"unseen acc={te:.3f}")
    print("=> keep RULES in weights, push FACTS to the external store.")
    return rows


if __name__ == "__main__":
    import sys
    report(profile="tiny" if "--tiny" in sys.argv else "full")
