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

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
NPZ = "data/comments_relating_sycophan_gemini_d131072_sparse_activations.npz"
JSONL = "data/r_chatGPT_comments_relating_sycophan_2022-12-22_2026-8-7.jsonl"

BATCH_SIZE = 20
DEFAULT_CONCURRENCY = 20
DEFAULT_REPEATS = 2
MAX_RETRIES = 3
MAX_CHARS = 4000  # per-comment truncation for the labelling prompt; generous cap, not a tight
                  # limit -- 600 was silently cutting real comments mid-argument (a ~950-char
                  # comment lost its own conclusion), and a 20-doc batch at 4000 chars/doc is
                  # trivially inside Haiku's context window at negligible extra cost.


def _is_fatal_api_error(e):
    """Billing/auth/permission errors won't resolve on retry -- fail immediately instead of
    burning 3 retries per in-flight call and then silently treating a whole outage as 'no-match'."""
    return isinstance(e, anthropic.APIStatusError) and e.status_code in (400, 401, 403)


def _retry_delay(e, attempt):
    """Respect the API's own Retry-After on 429s instead of blind exponential backoff -- under
    the higher concurrency needed for --repeats > 1, rate-limit throttling is expected, and
    guessing a backoff instead of reading Retry-After risks compounding it well past the
    run's time budget."""
    if isinstance(e, anthropic.APIStatusError) and e.status_code == 429:
        retry_after = e.response.headers.get("retry-after") if e.response is not None else None
        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass
    return 2 ** attempt

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
Read the ENTIRE comment before deciding, not just the part that looks relevant. A cherry-picked
fragment read in isolation does not count -- decide using the full comment as context.
If another clause elsewhere in the SAME comment contradicts, qualifies, or changes the meaning of
the part that would otherwise support a match, ask: does that other clause SUBSTANTIALLY NEGATE
the specific claim named in the criterion (e.g. the comment asserts the literal opposite, reveals
the supporting part was hypothetical or sarcastic, or attributes it to someone else's view)? If
so, the comment does NOT satisfy the criterion. But if the other clause is a separate aside,
hedge, or self-directed question that does not actually undermine the specific claim in the
criterion, it does not disqualify an otherwise genuine match -- do not require the whole comment
to be free of any qualification anywhere in it.
First, quote the exact span that most directly supports a match, or write null if none exists.
Then decide: does the comment as a whole actually satisfy the criterion as written?
If the verdict is "no" and a quote exists, briefly say why the quote doesn't count (e.g. "negated
by the next sentence", "hypothetical", "attributed to someone else").
Return ONLY JSON: {"quote": "<verbatim span or null>", "match": "yes"|"no", "reason": "<short phrase, only when match is no and a quote exists, else null>"}"""


def _seed_examples_block(fact, max_examples=2):
    """Render up to max_examples of a fact's seed_examples (full-text quotes the user cited when
    manually writing the fact) as concrete worked examples, anchoring the model on what a true
    positive actually looks like instead of only ever seeing the abstract evidence_criterion
    sentence."""
    examples = (fact.get("seed_examples") or [])[:max_examples]
    if not examples:
        return ""
    rendered = "\n\n".join(
        f'[doc_id={ex["doc_id"]}] {ex["text"]}\n-> this counts as a match.' for ex in examples
    )
    return f"\n\nEXAMPLE(S) OF A TRUE POSITIVE:\n{rendered}"


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

async def label_all(client, sem, model, fact, docs):
    """Label every doc against one fact's criterion, batched and concurrent. Missing/failed labels count as 'no'.

    If the fact sets "extract_field", a "yes" label also carries that field (e.g. which named
    trait a comment invoked), returned as {"match": "yes"|"no", "<field>": ...} per doc_id instead
    of a plain "yes"/"no" string.

    `sem` is a semaphore shared across every fact/repeat in the run (not created here), so
    --concurrency is an actual global cap on in-flight API calls rather than a per-call-site one.
    """
    criterion = fact["evidence_criterion"]
    extract_field = fact.get("extract_field")
    seed_block = _seed_examples_block(fact)
    if extract_field:
        field_description = fact.get("extract_field_description", extract_field)
        sys_msg = (LABEL_SYSTEM_EXTRACT.format(field=extract_field, field_description=field_description)
                   + f"\n\nCRITERION: {criterion}" + seed_block)
    else:
        sys_msg = f"{LABEL_SYSTEM}\n\nCRITERION: {criterion}" + seed_block
    batches = [docs[i : i + BATCH_SIZE] for i in range(0, len(docs), BATCH_SIZE)]

    async def one(batch):
        body = "\n".join(f'[doc_id={d["doc_id"]}] {d["text"][:MAX_CHARS]}' for d in batch)
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    r = await client.messages.create(
                        model=model, max_tokens=1500, temperature=0.0,
                        system=sys_msg, messages=[{"role": "user", "content": body}],
                    )
                    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                    m = re.search(r"\{.*\}", text, re.S)
                    result = json.loads(m.group(0)).get("labels", {}) if m else {}
                    missing = {d["doc_id"] for d in batch} - result.keys()
                    if missing:
                        print(f"  WARNING: fact {fact.get('fact_id')} batch response missing "
                              f"{len(missing)} doc_id(s), counted as 'no': {sorted(missing)}")
                    return result
                except Exception as e:  # noqa: BLE001
                    if _is_fatal_api_error(e) or attempt == MAX_RETRIES - 1:
                        raise RuntimeError(f"label batch permanently failed: {e}") from e
                    await asyncio.sleep(_retry_delay(e, attempt))

    results = await asyncio.gather(*[one(b) for b in batches])
    labels = {}
    for r in results:
        labels.update(r)
    return labels


async def verify_candidates(client, sem, model, fact, docs):
    """Individually re-verify each doc already labelled 'yes' by the batched pass, one doc per
    call, quote-grounded. Returns {doc_id: (is_hit: bool, quote: str|None, reason: str|None)}.

    `sem` is shared across every fact/repeat in the run, same as label_all."""
    criterion = fact["evidence_criterion"]
    sys_msg = f"{VERIFY_SYSTEM}\n\nCRITERION: {criterion}" + _seed_examples_block(fact)

    async def one(d):
        async with sem:
            for attempt in range(MAX_RETRIES):
                try:
                    r = await client.messages.create(
                        model=model, max_tokens=300, temperature=0.0,
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
                        obj = {"match": mm.group(1) if mm else "no", "quote": None, "reason": None}
                    match = str(obj.get("match", "no")).strip().lower().startswith("y")
                    return d["doc_id"], (match, obj.get("quote"), obj.get("reason"))
                except Exception as e:  # noqa: BLE001
                    if _is_fatal_api_error(e) or attempt == MAX_RETRIES - 1:
                        raise RuntimeError(f"verify call permanently failed for {d['doc_id']}: {e}") from e
                    await asyncio.sleep(_retry_delay(e, attempt))

    results = await asyncio.gather(*[one(d) for d in docs])
    return dict(results)


def _apply_verification(labels, verify_map):
    """Overlay verify_candidates results onto the batched labels dict, downgrading any candidate
    that failed isolated re-verification back to 'no' and attaching the grounding quote/reason."""
    out = dict(labels)
    for doc_id, (is_hit, quote, reason) in verify_map.items():
        entry = out.get(doc_id, "")
        entry = dict(entry) if isinstance(entry, dict) else {"match": entry}
        entry["match"] = "yes" if is_hit else "no"
        entry["verify_quote"] = quote
        entry["verify_reason"] = reason
        out[doc_id] = entry
    return out


def _label_entry(labels, doc_id):
    """A label is either a plain 'yes'/'no' string, or (for extract_field facts) a
    {'match': 'yes'|'no', '<field>': ...} dict. Normalize access to both shapes."""
    return labels.get(str(doc_id), "")


def _is_hit(entry):
    v = entry.get("match", "") if isinstance(entry, dict) else entry
    return str(v).strip().lower().startswith("y")


def aggregate_votes(per_repeat_labels, docs):
    """Majority-vote each doc's per-repeat label entries into one aggregated entry:
    {"match": "yes"/"no", "votes_yes": int, "votes_total": int, plus any free-text fields
    (extract_field value, verify_quote, verify_reason) copied from a representative "yes"-voting
    repeat, since free text can't itself be voted on -- only match/no-match can}.

    Ties count as "no": with --repeats 2 (the default) this means both repeats must agree a doc
    is a hit. That's a deliberate bias toward the more reproducible reading rather than inflating
    hits with whichever repeat happened to say yes -- the whole point of voting is to stop a
    fact's pass/fail from hinging on a single run's API-level noise (observed: 13/22/25 hits for
    the identical fact/pool/temperature=0 across different single-shot runs)."""
    n = len(per_repeat_labels)
    out = {}
    for d in docs:
        doc_id = str(d["doc_id"])
        entries = [_label_entry(labels, doc_id) for labels in per_repeat_labels]
        hits = [_is_hit(e) for e in entries]
        votes_yes = sum(hits)
        agg = {"match": "yes" if votes_yes * 2 > n else "no", "votes_yes": votes_yes, "votes_total": n}
        # verify_quote/verify_reason explain a "no" verdict just as much as a "yes" one (that's
        # the whole point of demoted_examples), so copy them from any repeat that set them,
        # regardless of that repeat's own vote -- unlike extract_field, which is only meaningful
        # attached to a "yes".
        for e in entries:
            if isinstance(e, dict) and (e.get("verify_quote") is not None or e.get("verify_reason") is not None):
                agg.setdefault("verify_quote", e.get("verify_quote"))
                agg.setdefault("verify_reason", e.get("verify_reason"))
                break
        for e, hit in zip(entries, hits):
            if hit and isinstance(e, dict):
                agg.update({k: v for k, v in e.items() if k not in ("match", "verify_quote", "verify_reason")})
                break
        out[doc_id] = agg
    return out


def evaluate_fact(fact, n_docs, pool, neg, raw_votes, final_votes, floor_rate, floor_min):
    """raw_votes/final_votes are aggregate_votes() outputs over the same docs, from before and
    after verification respectively -- both vote-aggregated across all --repeats runs."""
    extract_field = fact.get("extract_field")
    hit_in = lambda d, votes: _is_hit(votes.get(str(d["doc_id"]), ""))
    pool_hits = [d for d in pool if hit_in(d, final_votes)]
    neg_hits = [d for d in neg if hit_in(d, final_votes)]
    demoted = [d for d in pool if hit_in(d, raw_votes) and not hit_in(d, final_votes)]

    floor = max(floor_min, math.ceil(floor_rate * n_docs))
    pool_rate = len(pool_hits) / max(len(pool), 1)
    neg_rate = len(neg_hits) / max(len(neg), 1)
    precision = pool_rate / (pool_rate + neg_rate) if (pool_rate + neg_rate) else 0.0

    def as_example(d):
        ex = {k: d[k] for k in ("doc_id", "text", "activation")}
        entry = final_votes.get(str(d["doc_id"]), {})
        if extract_field:
            ex[extract_field] = entry.get(extract_field)
        if entry.get("verify_quote") is not None:
            ex["verify_quote"] = entry.get("verify_quote")
        ex["votes_yes"] = entry.get("votes_yes")
        ex["votes_total"] = entry.get("votes_total")
        return ex

    def as_demoted_example(d):
        raw_entry = raw_votes.get(str(d["doc_id"]), {})
        final_entry = final_votes.get(str(d["doc_id"]), {})
        return {
            "doc_id": d["doc_id"], "text": d["text"], "activation": d["activation"],
            "pre_verify_votes_yes": raw_entry.get("votes_yes"),
            "pre_verify_votes_total": raw_entry.get("votes_total"),
            "post_verify_votes_yes": final_entry.get("votes_yes"),
            "post_verify_votes_total": final_entry.get("votes_total"),
            "verify_quote": final_entry.get("verify_quote"),
            "verify_reason": final_entry.get("verify_reason"),
        }

    return {
        "fact_id": fact.get("fact_id"),
        "stylized_fact": fact["stylized_fact"],
        "evidence_criterion": fact["evidence_criterion"],
        "n_docs": n_docs,
        "floor": floor,
        "pool_size": len(pool),
        "n_hits": len(pool_hits),
        "pre_verify_n_hits": sum(1 for d in pool if hit_in(d, raw_votes)),
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
        "demoted_examples": [
            as_demoted_example(d)
            for d in sorted(demoted, key=lambda d: -d["activation"])[: floor + 5]
        ],
    }


async def label_and_verify_once(client, sem, model, fact, pool, neg, skip_verify):
    """One full label(+verify) pass for one fact. Returns (raw_labels, final_labels) -- kept
    separate (rather than only a pre-verify count) so repeated calls can be majority-voted at
    both stages via aggregate_votes, and so a doc verify demotes stays auditable instead of
    vanishing once _apply_verification overwrites its label."""
    raw_labels = await label_all(client, sem, model, fact, pool + neg)
    if skip_verify:
        return raw_labels, raw_labels
    candidates = [d for d in (pool + neg) if _is_hit(_label_entry(raw_labels, d["doc_id"]))]
    verify_map = await verify_candidates(client, sem, model, fact, candidates)
    final_labels = _apply_verification(raw_labels, verify_map)
    return raw_labels, final_labels


def _recall_report(fact_id, seed_examples, raw_votes, final_votes, pool_ids):
    """Report whether each of a fact's manually-cited seed doc_ids was recovered as a final hit,
    and if not, at which stage it was lost -- distinct from hit_examples/demoted_examples, which
    are capped at floor+5 and would misreport a seed doc as 'missing' if it were a real hit that
    simply didn't make the truncated example list."""
    seed_ids = [ex["doc_id"] for ex in seed_examples]
    recovered = [sid for sid in seed_ids if _is_hit(final_votes.get(str(sid), ""))]
    lines = [f'  {fact_id:<5} recovered {len(recovered)}/{len(seed_ids)} seed example(s)']
    for sid in seed_ids:
        if sid in recovered:
            continue
        if sid not in pool_ids:
            where = "outside this run's top-decile pool/--max_pool cap -- not evaluated"
        elif _is_hit(raw_votes.get(str(sid), "")):
            where = "demoted by verification"
        else:
            where = "did not get a 'yes' from the batch-label pass"
        lines.append(f"         MISSING {sid}: {where}")
    return lines


async def run(args):
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    n_docs, pool, neg = build_pools(args.neuron, args.n_neg, args.max_pool, args.seed)
    pool_ids = {d["doc_id"] for d in pool}
    print(f"neuron {args.neuron}: {n_docs} active docs, pool={len(pool)} "
          f"(top-decile capped at {args.max_pool}), neg={len(neg)}")
    print(f"model={args.model}  repeats={args.repeats}  concurrency={args.concurrency}\n")

    facts = json.load(open(args.facts_file))
    sem = asyncio.Semaphore(args.concurrency)

    # Fan every fact's every repeat out together -- not one fact at a time -- so the shared
    # semaphore above, not a sequential per-fact loop, is what governs how fast the run goes.
    jobs = [(fi, label_and_verify_once(client, sem, args.model, fact, pool, neg, args.skip_verify))
            for fi, fact in enumerate(facts) for _ in range(args.repeats)]
    outcomes = await asyncio.gather(*(job for _, job in jobs))

    per_fact_raw = [[] for _ in facts]
    per_fact_final = [[] for _ in facts]
    for (fi, _), (raw_labels, final_labels) in zip(jobs, outcomes):
        per_fact_raw[fi].append(raw_labels)
        per_fact_final[fi].append(final_labels)

    results = []
    recall_lines = []
    for fi, fact in enumerate(facts):
        raw_votes = aggregate_votes(per_fact_raw[fi], pool + neg)
        final_votes = aggregate_votes(per_fact_final[fi], pool + neg)
        res = evaluate_fact(fact, n_docs, pool, neg, raw_votes, final_votes, args.floor_rate, args.floor_min)
        res["model"] = args.model
        res["repeats"] = args.repeats
        results.append(res)
        flag = "PASS" if res["passes_floor"] else "FAIL"
        print(f'  {res["fact_id"]:<5} {flag:<5} hits {res["n_hits"]:>3}/{res["floor"]:<3} '
              f'(pre-verify {res["pre_verify_n_hits"]:>3})  '
              f'pool_rate {res["pool_rate"]:.1%}  neg_rate {res["neg_rate"]:.1%}  '
              f'precision {res["precision"]:.2f}')
        if fact.get("seed_examples"):
            recall_lines += _recall_report(fact.get("fact_id"), fact["seed_examples"],
                                            raw_votes, final_votes, pool_ids)

    out_path = os.path.join(os.path.dirname(args.facts_file), "labelled_facts.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")

    if recall_lines:
        print("\nrecall against manually-cited seed examples:")
        print("\n".join(recall_lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--neuron", type=int, required=True)
    p.add_argument("--facts_file", type=str, required=True,
                    help='JSON list of {"fact_id", "stylized_fact", "evidence_criterion", '
                         '"extract_field" (optional), "extract_field_description" (optional), '
                         '"seed_examples" (optional, [{"doc_id", "text"}]) -- full-text quotes '
                         'the fact was originally written from; used as few-shot grounding in '
                         'the label/verify prompts and as a recall sanity-check at the end}')
    p.add_argument("--n_neg", type=int, default=120, help="Size of the negative/contrast pool.")
    p.add_argument("--max_pool", type=int, default=300,
                    help="Cap on the top-decile pool size, to bound API cost on high-n_docs neurons.")
    p.add_argument("--floor_rate", type=float, default=0.01,
                    help="Required hit rate over n_docs (0.01 == 10%% of the top-decile pool).")
    p.add_argument("--floor_min", type=int, default=5, help="Absolute minimum hits regardless of rate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help="Anthropic model id used for both the label and verify calls.")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                    help="Independent label(+verify) passes per fact, majority-voted per doc "
                         "(ties count as 'no'). Fixes run-to-run hit-count instability observed "
                         "even at temperature=0.0 (13/22/25 hits for one fact across runs); "
                         "--repeats 1 reproduces the old single-shot behavior.")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="Max in-flight API calls, shared across every fact and repeat in the "
                         "run (not per-fact as before). Raising --repeats multiplies call volume "
                         "roughly linearly, so this may need tuning up against your account's "
                         "actual rate-limit tier to keep wall-clock time down.")
    p.add_argument("--skip_verify", action="store_true",
                    help="Skip the individual quote-grounded re-verification pass over batch-pass "
                         "'yes' hits (faster/cheaper, but batching lets a genuine match's 'yes' "
                         "bleed onto an unrelated neighbor a few slots away -- measured at ~46% "
                         "false-positive rate on batch-only hits across an audited sample).")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
