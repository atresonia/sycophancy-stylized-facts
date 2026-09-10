"""Regenerate the two cross-neuron stylized-facts rollups from data/<neuron>/labelled_facts.json.

Replaces what was previously a hand-maintained process: reads every neuron's labelled_facts.json
(pass or fail -- label_facts.py already writes both), classifies each fact PASS / NEAR-MISS / FAIL,
and writes:
  - data/stylized_facts_consolidated.md -- passing facts (top-3 truncated quotes), plus a
    "Not yet passing (for review)" section per neuron so a fact that has real evidence but missed
    the floor (e.g. due to verify demotions or run-to-run noise) isn't silently invisible.
  - data/stylized_facts_evidence.md -- same, with top-10 untruncated quotes, plus any
    verify-demoted hits for facts that aren't (yet) passing.

Usage:
    python src/generate_rollups.py
"""
import argparse
import json
import os
import re

NEURON_LINE_RE = re.compile(r"NEURON\s+(\d+)\s*[-—]\s*(.+)")


def load_neuron_descriptions(path):
    """Parse data/neuron_descriptions_manual.md's "N. NEURON <id> - <description>" lines
    (bolded or not, hyphen or em-dash) into {neuron_id: description}."""
    descriptions = {}
    if not os.path.exists(path):
        return descriptions
    for raw in open(path):
        line = raw.strip().strip("*")
        line = re.sub(r"^\d+\.\s*", "", line)
        m = NEURON_LINE_RE.search(line)
        if m:
            descriptions[m.group(1)] = m.group(2).strip().rstrip("*").strip()
    return descriptions


def classify(fact, near_miss_hit_rate, near_miss_precision):
    if fact.get("passes_floor"):
        return "PASS"
    floor = fact.get("floor") or 0
    hit_rate = fact.get("n_hits", 0) / floor if floor else 0.0
    if hit_rate >= near_miss_hit_rate and fact.get("precision", 0.0) >= near_miss_precision:
        return "NEAR-MISS"
    return "FAIL"


def _collapse_ws(text):
    """Comment bodies carry literal newlines (Reddit paragraph breaks); render quotes as a single
    line so a multi-paragraph comment doesn't fracture the markdown list structure."""
    return " ".join(text.split())


def _truncate(text, max_chars):
    text = _collapse_ws(text)
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


NO_PASS_NOTE = (
    "_No candidate fact for this neuron cleared the precision/coverage bar after independent "
    "quote-grounded verification -- candidate facts need revisiting._"
)


def render_consolidated_neuron(neuron_id, desc, facts, near_miss_hit_rate, near_miss_precision,
                                max_quotes=3, quote_chars=300):
    lines = [f"## Neuron {neuron_id}: {desc}".rstrip(": "), ""]
    tags = {f["fact_id"]: classify(f, near_miss_hit_rate, near_miss_precision) for f in facts}
    passing = [f for f in facts if tags[f["fact_id"]] == "PASS"]

    if not passing:
        lines += [NO_PASS_NOTE, ""]
    else:
        for i, fact in enumerate(passing, 1):
            lines.append(f"{i}. {fact['stylized_fact']}")
            for letter, ex in zip("abcdefghijklmnopqrstuvwxyz", fact.get("hit_examples", [])[:max_quotes]):
                lines.append(f'   {letter}. ({ex["doc_id"]}) "{_truncate(ex["text"], quote_chars)}"')
            lines.append("")

    not_passing = [f for f in facts if tags[f["fact_id"]] != "PASS"]
    if not_passing:
        lines += ["### Not yet passing (for review)", ""]
        for fact in not_passing:
            tag = tags[fact["fact_id"]]
            lines.append(f'- [{tag}] {fact["stylized_fact"]} '
                          f'(fact_id={fact["fact_id"]}, hits {fact.get("n_hits")}/{fact.get("floor")}, '
                          f'precision={fact.get("precision")})')
            for ex in (fact.get("demoted_examples") or [])[:2]:
                reason = f' -- verify: "{ex["verify_reason"]}"' if ex.get("verify_reason") else ""
                lines.append(f'   - demoted: ({ex["doc_id"]}) "{_truncate(ex["text"], quote_chars)}"{reason}')
        lines.append("")

    return lines


def render_evidence_neuron(neuron_id, desc, facts, near_miss_hit_rate, near_miss_precision,
                            max_quotes=10):
    lines = [f"## Neuron {neuron_id}: {desc}".rstrip(": "), ""]
    tags = {f["fact_id"]: classify(f, near_miss_hit_rate, near_miss_precision) for f in facts}
    passing = [f for f in facts if tags[f["fact_id"]] == "PASS"]

    if not passing:
        lines += [NO_PASS_NOTE, ""]
    else:
        for i, fact in enumerate(passing, 1):
            lines.append(f"### {i}. {fact['stylized_fact']}")
            lines.append(f'*fact_id={fact["fact_id"]}, n_hits={fact.get("n_hits")}/{fact.get("floor")}, '
                         f'precision={fact.get("precision")}*')
            lines.append("")
            for j, ex in enumerate(fact.get("hit_examples", [])[:max_quotes], 1):
                lines.append(f'{j}. ({ex["doc_id"]}, activation={ex["activation"]:.3f}) "{_collapse_ws(ex["text"])}"')
            lines.append("")

    not_passing = [f for f in facts if tags[f["fact_id"]] != "PASS"]
    if not_passing:
        lines += ["### Not yet passing (for review)", ""]
        for fact in not_passing:
            tag = tags[fact["fact_id"]]
            lines.append(f'#### [{tag}] {fact["stylized_fact"]}')
            lines.append(f'*fact_id={fact["fact_id"]}, n_hits={fact.get("n_hits")}/{fact.get("floor")}, '
                         f'precision={fact.get("precision")}*')
            lines.append("")
            demoted = fact.get("demoted_examples") or []
            if demoted:
                lines.append("Demoted examples (batch pass said yes, verification said no):")
                lines.append("")
                for ex in demoted:
                    quote = f' verify quote: "{ex["verify_quote"]}"' if ex.get("verify_quote") else ""
                    reason = f' ({ex["verify_reason"]})' if ex.get("verify_reason") else ""
                    lines.append(f'- ({ex["doc_id"]}, activation={ex["activation"]:.3f}) '
                                  f'"{_collapse_ws(ex["text"])}"{quote}{reason}')
                lines.append("")

    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--near_miss_hit_rate", type=float, default=0.9,
                     help="A fact that isn't passing but has n_hits >= this fraction of its floor "
                          "(and clears --near_miss_precision) is labelled NEAR-MISS instead of FAIL.")
    ap.add_argument("--near_miss_precision", type=float, default=0.55)
    args = ap.parse_args()

    descriptions = load_neuron_descriptions(os.path.join(args.data_dir, "neuron_descriptions_manual.md"))
    neuron_ids = sorted(
        (d for d in os.listdir(args.data_dir)
         if d.isdigit() and os.path.exists(os.path.join(args.data_dir, d, "labelled_facts.json"))),
        key=int,
    )

    consolidated = [
        "# Consolidated Stylized Facts", "",
        "_Hits are each individually re-verified (quote-grounded, isolated from the batch) before "
        "counting; includes facts passing the floor after verification, plus near-misses with hits "
        f"≥{int(args.near_miss_hit_rate * 100)}% of floor and precision ≥"
        f"{args.near_miss_precision}. Top 3 quotes shown per fact (truncated); facts not yet "
        "passing are listed separately per neuron for review, not omitted. See "
        "[stylized_facts_evidence.md](stylized_facts_evidence.md) for top 10, untruncated._", "",
    ]
    evidence = [
        "# Stylized Facts — Full Evidence", "",
        "_Top 10 hit examples per fact, sorted by activation, untruncated. Facts not yet passing "
        "are listed separately per neuron, along with any verification-demoted hits, for review. "
        "Companion to [stylized_facts_consolidated.md](stylized_facts_consolidated.md)._", "",
    ]

    for neuron_id in neuron_ids:
        facts = json.load(open(os.path.join(args.data_dir, neuron_id, "labelled_facts.json")))
        desc = descriptions.get(neuron_id, "")
        consolidated += render_consolidated_neuron(
            neuron_id, desc, facts, args.near_miss_hit_rate, args.near_miss_precision)
        consolidated.append("")
        evidence += render_evidence_neuron(
            neuron_id, desc, facts, args.near_miss_hit_rate, args.near_miss_precision)
        evidence.append("")

    out_consolidated = os.path.join(args.data_dir, "stylized_facts_consolidated.md")
    out_evidence = os.path.join(args.data_dir, "stylized_facts_evidence.md")
    with open(out_consolidated, "w") as f:
        f.write("\n".join(consolidated).rstrip() + "\n")
    with open(out_evidence, "w") as f:
        f.write("\n".join(evidence).rstrip() + "\n")
    print(f"wrote {out_consolidated} and {out_evidence} for {len(neuron_ids)} neuron(s)")


if __name__ == "__main__":
    main()
