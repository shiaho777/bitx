import json
import os
from typing import List


def make_suite(size: int) -> List[dict]:
    if size < 6:
        raise ValueError("suite size must be at least 6")
    rows = []
    for i in range(size):
        edit = i < max(4, size // 4)
        delete = i == max(4, size // 4) - 1
        rows.append({
            "id": f"entity_{i:05d}",
            "prompt": f"attribute of entity {i:05d}",
            "paraphrase": f"entity {i:05d} attribute",
            "old": f"value_{i:05d}_old",
            "new": f"value_{i:05d}_new" if edit else None,
            "edit": edit,
            "delete": delete,
        })
    return rows


def make_keyed_suite(size: int) -> List[dict]:
    if size < 6:
        raise ValueError("suite size must be at least 6")
    rows = []
    for i in range(size):
        edit = i < max(4, size // 4)
        delete = i == max(4, size // 4) - 1
        key = keyed_name(i)
        rows.append({
            "id": key,
            "prompt": f"registry key {key} primary attribute",
            "paraphrase": f"primary attribute for registry key {key}",
            "old": f"{key}_old_value",
            "new": f"{key}_new_value" if edit else None,
            "edit": edit,
            "delete": delete,
        })
    return rows


def make_heldout_ambiguity_suite(size: int, n_domains: int = 24, seed: int = 100) -> List[dict]:
    """Phase 1: Generate a larger held-out semantic ambiguity suite.

    Unlike the small 12-scenario default suite, this generates a larger
    set with diverse domain pairs, varying query templates, and explicit
    held-out partition markers for train/test split evaluation.

    Each scenario has the standard ambiguity contract plus:
    - partition: "train" or "heldout"
    - query_variants: 2-3 natural language phrasings per scenario
    """
    import random

    if size < 6:
        raise ValueError("held-out ambiguity suite size must be at least 6")
    domains = ambiguity_domains()
    rng = random.Random(seed)
    rows = []
    n_heldout = size // 3  # 1/3 held out
    for i in range(size):
        right = domains[rng.randint(0, len(domains) - 1)]
        wrong = domains[rng.randint(0, len(domains) - 1)]
        while wrong["domain"] == right["domain"]:
            wrong = domains[rng.randint(0, len(domains) - 1)]
        distractor = domains[rng.randint(0, len(domains) - 1)]
        partition = "heldout" if i >= size - n_heldout else "train"
        query_templates = [
            "Which policy applies to the request?",
            "Where should this case be handled?",
            "Who owns the next step for this issue?",
            "What process covers this update?",
            "How should the record be routed?",
            "Which instructions should be followed here?",
            "Where do we escalate this item?",
            "What rules govern this change?",
        ]
        anchors = [
            right["topic"], wrong["topic"], distractor["topic"],
            "the shared request", "the ambiguous item",
        ]
        base_query = f"{query_templates[i % len(query_templates)]} Context: {anchors[i % len(anchors)]}."
        variants = [
            base_query,
            f"Help me figure out {anchors[(i+1) % len(anchors)]}.",
            f"I need guidance on {anchors[(i+2) % len(anchors)]}.",
        ]
        rows.append({
            "id": f"heldout_amb_{i:04d}_{right['domain']}_vs_{wrong['domain']}",
            "partition": partition,
            "query_text": base_query,
            "query_variants": variants,
            "right_text": right["text"],
            "right_value": right["value"],
            "wrong_text": wrong["text"],
            "wrong_value": wrong["value"],
            "clarify": f"clarify: need {right['domain']} or {wrong['domain']} domain",
            "right_domain": right["domain"],
            "wrong_domain": wrong["domain"],
        })
    return rows


def make_ambiguity_suite(size: int) -> List[dict]:
    if size < 6:
        raise ValueError("suite size must be at least 6")
    domains = ambiguity_domains()
    rows = []
    for i in range(size):
        right = domains[(i * 5 + 1) % len(domains)]
        wrong = domains[(i * 7 + 4) % len(domains)]
        if right["domain"] == wrong["domain"]:
            wrong = domains[(i * 7 + 5) % len(domains)]
        distractor = domains[(i * 11 + 9) % len(domains)]
        rows.append({
            "id": f"ambiguous_{i:05d}_{right['domain']}_vs_{wrong['domain']}",
            "query_text": ambiguity_query(i, right, wrong, distractor),
            "right_text": right["text"],
            "right_value": right["value"],
            "wrong_text": wrong["text"],
            "wrong_value": wrong["value"],
            "clarify": f"clarify: need {right['domain']} or {wrong['domain']} domain",
        })
    return rows


def ambiguity_query(i: int, right: dict, wrong: dict, distractor: dict) -> str:
    templates = [
        "Which policy applies to the request?",
        "Where should this case be handled?",
        "Who owns the next step for this issue?",
        "What process covers this update?",
        "How should the record be routed?",
        "Which instructions should be followed here?",
        "Where do we escalate this item?",
        "What rules govern this change?",
    ]
    anchors = [
        right["topic"],
        wrong["topic"],
        distractor["topic"],
        "the shared request",
        "the ambiguous item",
    ]
    return f"{templates[i % len(templates)]} Context: {anchors[i % len(anchors)]}."


def ambiguity_domains() -> List[dict]:
    return [
        {"domain": "clinic", "topic": "visitor appointment", "text": "clinic scheduling instructions guests", "value": "visitor clinic"},
        {"domain": "shipping", "topic": "parcel handoff", "text": "warehouse shipping manifest protocol", "value": "shipping desk"},
        {"domain": "garden", "topic": "seedling care", "text": "garden irrigation maintenance guide", "value": "garden watering"},
        {"domain": "finance", "topic": "late invoice", "text": "finance billing arrears procedure", "value": "billing arrears"},
        {"domain": "identity", "topic": "locked account", "text": "identity credentials recovery workflow", "value": "account recovery"},
        {"domain": "property", "topic": "roof damage", "text": "property repairs claim intake", "value": "property claim"},
        {"domain": "nutrition", "topic": "allergy meal", "text": "nutrition dietary restriction menu", "value": "allergy menu"},
        {"domain": "software", "topic": "security patch", "text": "software vulnerability release register", "value": "security patch"},
        {"domain": "recruiting", "topic": "offer letter", "text": "human resources recruiting authorization", "value": "recruiting approval"},
        {"domain": "laboratory", "topic": "sample custody", "text": "laboratory specimen chain custody", "value": "specimen custody"},
        {"domain": "leasing", "topic": "tenant deposit", "text": "leasing security escrow procedure", "value": "tenant escrow"},
        {"domain": "operations", "topic": "service outage", "text": "operations incident response channel", "value": "incident response"},
        {"domain": "facilities", "topic": "building access", "text": "facilities maintenance access schedule", "value": "facility access"},
        {"domain": "legal", "topic": "contract exception", "text": "legal contract exception review", "value": "contract review"},
        {"domain": "compliance", "topic": "audit evidence", "text": "compliance audit evidence retention", "value": "audit retention"},
        {"domain": "support", "topic": "customer escalation", "text": "support customer escalation runbook", "value": "support escalation"},
        {"domain": "inventory", "topic": "stock adjustment", "text": "inventory stock adjustment ledger", "value": "stock ledger"},
        {"domain": "security", "topic": "badge incident", "text": "security badge incident protocol", "value": "badge incident"},
        {"domain": "travel", "topic": "trip reimbursement", "text": "travel reimbursement approval guide", "value": "travel approval"},
        {"domain": "training", "topic": "course enrollment", "text": "training course enrollment policy", "value": "course enrollment"},
        {"domain": "procurement", "topic": "vendor onboarding", "text": "procurement vendor onboarding checklist", "value": "vendor onboarding"},
        {"domain": "payroll", "topic": "timesheet correction", "text": "payroll timesheet correction workflow", "value": "timesheet correction"},
        {"domain": "manufacturing", "topic": "line stoppage", "text": "manufacturing line stoppage response", "value": "line response"},
        {"domain": "research", "topic": "experiment protocol", "text": "research experiment protocol registry", "value": "experiment registry"},
    ]


def keyed_name(i: int) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    a = alphabet[i % len(alphabet)]
    b = alphabet[(i // len(alphabet)) % len(alphabet)]
    c = alphabet[(i // (len(alphabet) * len(alphabet))) % len(alphabet)]
    return f"bx-{a}{b}{c}-{i:05d}"


def write_suite(path: str, rows: List[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_suite(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
