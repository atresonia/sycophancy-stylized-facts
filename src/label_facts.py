"""Mechanically apply a candidate fact's evidence_criterion across a wide pool of a neuron's
activations, to check support against a floor before spending a human blind-labelling pass on it.

Floor rationale: n_docs varies ~24x across candidate neurons, so a flat example count isn't
comparable across neurons. floor = max(floor_min, ceil(floor_rate * n_docs)) with floor_rate=0.01
is equivalent to requiring the fact to hold in >=10% of the neuron's top-decile pool -- the same
population used for interpretation, so it's not an arbitrary cut.

Usage:
    python src/label_facts.py --neuron 39774 --facts_file data/39774/candidate_facts.json
"""
import argparse
import asyncio
import json
import math
import os
import re

import numpy as np
import scipy.sparse as sp
import anthropic
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-4-5-20251001"
NPZ = "data/comments_relating_sycophan_gemini_d131072_sparse_activations.npz"
JSONL = "data/r_chatGPT_comments_relating_sycophan_2022-12-22_2026-8-7.jsonl"

BATCH_SIZE = 20
CONCURRENCY = 8
MAX_RETRIES = 3
MAX_CHARS = 600  # per-comment truncation for the labelling prompt


def _is_fatal_api_error(e):
    """Billing/auth/permission errors won't resolve on retry -- fail immediately instead of
    burning 3 retries per in-flight call and then silently treating a whole outage as 'no-match'."""
    return isinstance(e, anthropic.APIStatusError) and e.status_code in (400, 401, 403)

LABEL_SYSTEM = """You check whether short Reddit comments satisfy a specific criterion.
For each comment, answer "yes" only if it satisfies the criterion as literally stated, else "no".
Do not infer intent beyond what the text says. A comment must actually instantiate the
criterion, not merely be on the same general topic as it.
Return ONLY a JSON object: {"labels": {"<doc_id>": "yes"|"no", ...}}
Include every doc_id you were given."""

LABEL_SYSTEM_EXTRACT = """You check whether short Reddit comments satisfy a specific criterion.
For each comment, decide "yes" only if it satisfies the criterion as literally stated, else "no".
Do not infer intent beyond what the text says. A comment must actually instantiate the
criterion, not merely be on the same general topic as it.
When (and only when) a comment is "yes", also extract {field}: {field_description}
Return ONLY a JSON object:
{{"labels": {{"<doc_id>": {{"match": "yes"|"no", "{field}": "<short value, or null if match is no>"}}, ...}}}}
Include every doc_id you were given."""

# A batched call asks the model for N independent "yes"/"no" verdicts in one response; in practice
# a genuine "yes" for one comment can bleed onto a topically-adjacent neighbor a few slots away in
# the same batch. VERIFY_SYSTEM re-checks each batch-pass "yes" alone, one doc per call, and forces
# a verbatim quote before the verdict so the model grounds its answer in that doc's actual text
# instead of the batch's general topic. Measured against a hand-audited sample this closed most of
# the gap between batch-labelled precision and true precision.
VERIFY_SYSTEM = """You check whether a short Reddit comment satisfies a specific criterion.
Read the ENTIRE comment before deciding, not just the part that looks relevant. If one clause
would support a match but another clause elsewhere in the SAME comment contradicts, qualifies, or
changes its meaning (e.g. the comment also asserts the opposite claim, reveals the supporting part
was hypothetical or sarcastic, or attributes it to someone else's view), the comment does NOT
satisfy the criterion -- a cherry-picked fragment read in isolation does not count.
First, quote the exact span that most directly supports a match, or write null if none exists.
Then decide, using the full comment as context (not the quoted span read in isolation, not the
comment's general topic, not something implied but unstated): does the comment as a whole actually
satisfy the criterion as written?
Return ONLY JSON: {"quote": "<verbatim span or null>", "match": "yes"|"no"}"""


# ---------------------------------------------------------------- data

def load_npz(path):
    d = np.load(path)
    N, F = int(d["shape"][0]), int(d["shape"][1])
    return sp.csr_matrix((d["values"].astype(np.float32),
                          (d["row_indices"], d["feature_indices"])),
                         shape=(N, F)).tocsc()


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def clean_data(data, min_char=40):
    bad = {"AutoModerator", "ChatGPT-ModTeam"}
    seen, m = set(), np.zeros(len(data), dtype=bool)
    for i, row in enumerate(data):
        body = row.get("body", "").strip()
        ok = (row.get("author") not in bad and len(body) >= min_char
              and body not in ("[removed]", "[deleted]", "") and body not in seen)
        m[i] = ok
        if ok:
            seen.add(body)
    return m


def build_pools(neuron, n_neg, max_pool, seed):
    """Top-decile pool (capped at max_pool) plus a keyword/random negative pool for the same neuron."""
    data = load_jsonl(JSONL)
    X = load_npz(NPZ)
    clean = clean_data(data)
    a = X[:, neuron].toarray().ravel()

    active = np.flatnonzero((a > 0) & clean)
    active = active[np.argsort(-a[active])]
    n_docs = len(active)
    decile = max(1, n_docs // 10)
    pool_idx = active[: min(decile, max_pool)]

    rng = np.random.default_rng(seed)
    inactive = np.flatnonzero((a == 0) & clean)
    kw = re.compile(r"sycophan", re.I)
    inact_kw = np.array([i for i in inactive if kw.search(data[i]["body"])], dtype=int)
    n_kw = min(n_neg // 2, len(inact_kw))
    neg_kw = rng.choice(inact_kw, n_kw, replace=False) if n_kw else np.array([], dtype=int)
    rest = np.setdiff1d(inactive, neg_kw)
    neg_rand = rng.choice(rest, min(n_neg - n_kw, len(rest)), replace=False)
    neg_idx = np.concatenate([neg_kw, neg_rand])

    as_doc = lambda i: {"doc_id": data[i]["id"], "text": data[i]["body"], "activation": float(a[i])}
    pool = [as_doc(i) for i in pool_idx]
    neg = [as_doc(i) for i in neg_idx]
    return n_docs, pool, neg


# ------------------------------------------------------------- labelling

async def label_all(client, fact, docs):
    """Label every doc against one fact's criterion, batched and concurrent. Missing/failed labels count as 'no'.

    If the fact sets "extract_field", a "yes" label also carries that field (e.g. which named
    trait a comment invoked), returned as {"match": "yes"|"no", "<field>": ...} per doc_id instead
    of a plain "yes"/"no" string.
    """
    criterion = fact["evidence_criterion"]
    extract_field = fact.get("extract_field")
    if extract_field:
        field_description = fact.get("extract_field_description", extract_field)
        sys_msg = (LABEL_SYSTEM_EXTRACT.format(field=extract_field, field_description=field_description)
                   + f"\n\nCRITERION: {criterion}")
    else:
        sys_msg = f"{LABEL_SYSTEM}\n\nCRITERION: {criterion}"
    batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(batch):
        body = "\n".join(f'[doc_id={d["doc_id"]}] {d["text"][:MAX_CHARS]}' for d in batch)
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    r = await client.messages.create(
                        model=MODEL, max_tokens=1500, temperature=0.0,
                        system=sys_msg, messages=[{"role": "user", "content": body}],
                    )
                    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                    m = re.search(r"\{.*\}", text, re.S)
                    return json.loads(m.group(0)).get("labels", {}) if m else {}
                except Exception as e:  # noqa: BLE001
                    if _is_fatal_api_error(e) or attempt == MAX_RETRIES - 1:
                        raise RuntimeError(f"label batch permanently failed: {e}") from e
                    await asyncio.sleep(2 ** attempt)

    results = await asyncio.gather(*[one(b) for b in batches])
    labels = {}
    for r in results:
        labels.update(r)
    return labels


async def verify_candidates(client, criterion, docs):
    """Individually re-verify each doc already labelled 'yes' by the batched pass, one doc per
    call, quote-grounded. Returns {doc_id: (is_hit: bool, quote: str|None)}."""
    sys_msg = f"{VERIFY_SYSTEM}\n\nCRITERION: {criterion}"
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(d):
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    r = await client.messages.create(
                        model=MODEL, max_tokens=300, temperature=0.0,
                        system=sys_msg, messages=[{"role": "user", "content": d["text"][:MAX_CHARS]}],
                    )
                    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                    m = re.search(r"\{.*\}", text, re.S)
                    try:
                        obj = json.loads(m.group(0)) if m else {}
                    except json.JSONDecodeError:
                        # the verbatim quote often contains unescaped quote marks from the
                        # source comment, breaking strict JSON; fall back to reading "match"
                        # directly since that's the field that actually decides the verdict.
                        mm = re.search(r'"match"\s*:\s*"(yes|no)"', text, re.I)
                        obj = {"match": mm.group(1) if mm else "no", "quote": None}
                    match = str(obj.get("match", "no")).strip().lower().startswith("y")
                    return d["doc_id"], (match, obj.get("quote"))
                except Exception as e:  # noqa: BLE001
                    if _is_fatal_api_error(e) or attempt == MAX_RETRIES - 1:
                        raise RuntimeError(f"verify call permanently failed for {d['doc_id']}: {e}") from e
                    await asyncio.sleep(2 ** attempt)

    results = await asyncio.gather(*[one(d) for d in docs])
    return dict(results)


def _apply_verification(labels, verify_map):
    """Overlay verify_candidates results onto the batched labels dict, downgrading any candidate
    that failed isolated re-verification back to 'no' and attaching the grounding quote."""
    out = dict(labels)
    for doc_id, (is_hit, quote) in verify_map.items():
        entry = out.get(doc_id, "")
        entry = dict(entry) if isinstance(entry, dict) else {"match": entry}
        entry["match"] = "yes" if is_hit else "no"
        entry["verify_quote"] = quote
        out[doc_id] = entry
    return out


def _label_entry(labels, doc_id):
    """A label is either a plain 'yes'/'no' string, or (for extract_field facts) a
    {'match': 'yes'|'no', '<field>': ...} dict. Normalize access to both shapes."""
    return labels.get(str(doc_id), "")


def _is_hit(entry):
    v = entry.get("match", "") if isinstance(entry, dict) else entry
    return str(v).strip().lower().startswith("y")


def evaluate_fact(fact, n_docs, pool, neg, labels, floor_rate, floor_min, pre_verify_pool_hits=None):
    extract_field = fact.get("extract_field")
    is_hit = lambda d: _is_hit(_label_entry(labels, d["doc_id"]))
    pool_hits = [d for d in pool if is_hit(d)]
    neg_hits = [d for d in neg if is_hit(d)]

    floor = max(floor_min, math.ceil(floor_rate * n_docs))
    pool_rate = len(pool_hits) / max(len(pool), 1)
    neg_rate = len(neg_hits) / max(len(neg), 1)
    precision = pool_rate / (pool_rate + neg_rate) if (pool_rate + neg_rate) else 0.0

    def as_example(d):
        ex = {k: d[k] for k in ("doc_id", "text", "activation")}
        entry = _label_entry(labels, d["doc_id"])
        if extract_field and isinstance(entry, dict):
            ex[extract_field] = entry.get(extract_field)
        if isinstance(entry, dict) and entry.get("verify_quote") is not None:
            ex["verify_quote"] = entry.get("verify_quote")
        return ex

    return {
        "fact_id": fact.get("fact_id"),
        "stylized_fact": fact["stylized_fact"],
        "evidence_criterion": fact["evidence_criterion"],
        "n_docs": n_docs,
        "floor": floor,
        "pool_size": len(pool),
        "n_hits": len(pool_hits),
        "pre_verify_n_hits": pre_verify_pool_hits,
        "pool_rate": round(pool_rate, 3),
        "neg_size": len(neg),
        "n_false_positives": len(neg_hits),
        "neg_rate": round(neg_rate, 3),
        "precision": round(precision, 3),
        "passes_floor": len(pool_hits) >= floor,
        "hit_examples": [
            as_example(d)
            for d in sorted(pool_hits, key=lambda d: -d["activation"])[: floor + 5]
        ],
    }


async def run(args):
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    n_docs, pool, neg = build_pools(args.neuron, args.n_neg, args.max_pool, args.seed)
    print(f"neuron {args.neuron}: {n_docs} active docs, pool={len(pool)} "
          f"(top-decile capped at {args.max_pool}), neg={len(neg)}\n")

    facts = json.load(open(args.facts_file))
    results = []
    for fact in facts:
        raw_labels = await label_all(client, fact, pool + neg)
        pre_verify_pool_hits = sum(1 for d in pool if _is_hit(_label_entry(raw_labels, d["doc_id"])))

        if args.skip_verify:
            labels = raw_labels
        else:
            candidates = [d for d in (pool + neg) if _is_hit(_label_entry(raw_labels, d["doc_id"]))]
            verify_map = await verify_candidates(client, fact["evidence_criterion"], candidates)
            labels = _apply_verification(raw_labels, verify_map)

        res = evaluate_fact(fact, n_docs, pool, neg, labels, args.floor_rate, args.floor_min,
                             pre_verify_pool_hits=pre_verify_pool_hits)
        results.append(res)
        flag = "PASS" if res["passes_floor"] else "FAIL"
        print(f'  {res["fact_id"]:<5} {flag:<5} hits {res["n_hits"]:>3}/{res["floor"]:<3} '
              f'(pre-verify {res["pre_verify_n_hits"]:>3})  '
              f'pool_rate {res["pool_rate"]:.1%}  neg_rate {res["neg_rate"]:.1%}  '
              f'precision {res["precision"]:.2f}')

    out_path = os.path.join(os.path.dirname(args.facts_file), "labelled_facts.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neuron", type=int, required=True)
    p.add_argument("--facts_file", type=str, required=True,
                    help='JSON list of {"fact_id", "stylized_fact", "evidence_criterion", '
                         '"extract_field" (optional), "extract_field_description" (optional)}')
    p.add_argument("--n_neg", type=int, default=120, help="Size of the negative/contrast pool.")
    p.add_argument("--max_pool", type=int, default=300,
                    help="Cap on the top-decile pool size, to bound API cost on high-n_docs neurons.")
    p.add_argument("--floor_rate", type=float, default=0.01,
                    help="Required hit rate over n_docs (0.01 == 10%% of the top-decile pool).")
    p.add_argument("--floor_min", type=int, default=5, help="Absolute minimum hits regardless of rate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_verify", action="store_true",
                    help="Skip the individual quote-grounded re-verification pass over batch-pass "
                         "'yes' hits (faster/cheaper, but batching lets a genuine match's 'yes' "
                         "bleed onto an unrelated neighbor a few slots away -- measured at ~46% "
                         "false-positive rate on batch-only hits across an audited sample).")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
