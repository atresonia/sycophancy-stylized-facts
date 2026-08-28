"""Generate "existence" stylized facts from a top_features_<target>.jsonl file
An "existence" stylized fact is a stylized fact that can be directly verified from the supported examples provided
(no further statistical analysis or data studying needed)

Steps:
1. GENERATE: show every candidate feature (filter on max_thread_share) with a sample of its top-activating
   comments (n_show_examples: default=25). The model writes N facts, each attributed to AT MOST K (1/5/10) features.
2. LABEL: take the K features from 1 and for each feature, pool the top-N (default=100) comments and label
   each fact with at most K * top-N comments (max: 100/500/1000).
3. REPAIR: revise the facts and examples generated, providing precision and other metrics.

Usage:
    python generate_stylized_facts.py --features_file data/features/top_features_gpt5_base.jsonl \
    --output_dir data/facts --n_features 5 \
    --target gpt5_marker
"""
import numpy as np
import hashlib
import os
import dotenv
import argparse
import collections
import datetime
import json
import random
import re
import sys
import time
import tqdm
import concurrent.futures as cf
from anthropic import Anthropic
from prompts import ATTR_SYSTEM, CORPUS_BLURB, GEN_SYSTEM, REPAIR_SYSTEM, TARGET_BLURBS

GEN_MODEL = "claude-sonnet-5"
ATTR_MODEL = "claude-haiku-4-5-20251001"
GEN_BUDGETS = [16_000, 32_000, 64_000]
MAX_TOKENS_ATTR = 4000
MAX_RETRIES = 5

# number of negative examples to provide for each feature
N_CONTRAST = 10
N_NEAR_MISS = 5 # non-matching pooled comments shown per fact at repair
N_FALSE_POS = 5 # false positives shown per fact at repair


def dedupe_features(features, embeddings_file, n_top, threshold):
    """Group features whose top-activating comments occupy the same region.
 
    Centering is not optional: these embeddings are strongly anisotropic, and on raw
    cosine every pair of features scores ~0.9 and the clustering collapses to one group.
    Returns one representative per cluster (highest |z|), each carrying the ids of the
    features it absorbed.
    """
    E = np.load(embeddings_file).astype(np.float32)
    E -= E.mean(0)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
 
    C = np.stack([E[[e["id"] for e in f["top"][:n_top]]].mean(0) for f in features])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    S = C @ C.T
    np.fill_diagonal(S, -1.0)
 
    order = sorted(range(len(features)), key=lambda i: -abs(features[i]["z"]))
    seen, reps = set(), []
    for i in order:
        if i in seen:
            continue
        members = [i] + [j for j in range(len(features))
                         if j not in seen and j != i and S[i, j] >= threshold]
        seen.update(members)
        rep = dict(features[i])
        rep["merged_feature_ids"] = [features[j]["neuron"] for j in members]
        reps.append(rep)
    print(f"deduped {len(features)} features -> {len(reps)} concepts "
          f"(threshold {threshold})")
    for r in reps:
        if len(r["merged_feature_ids"]) > 1:
            print(f"  {r['neuron']} absorbs {r['merged_feature_ids'][1:]}")
    return reps


def load_features_jsonl(features_file):
    with open(features_file, "r") as f:
        return [json.loads(line) for line in f]


def truncate_words(text, max_words):
    """Truncate text for optimizing for passing into LLMs"""
    if not max_words:
        return text
    w = text.split()
    return text if len(w) <= max_words else " ".join(w[:max_words]) + " [...]"


def as_doc(e, max_words):
    """Feature-file example -> labelling document."""
    return {"comment_id": e["id"], "body": e["body"], "created_utc": e["created_utc"],
            "created_utc_str": e["created_utc_str"],
            "prompt_body": truncate_words(e["body"], max_words), "sources": []}



def get_top_activations(features, n, max_words):
    """Get top-N activations for each feature
       Note: this removes duplicates: 
       ie: if a comment is the top-activating comment for multiple features, it will only be included once.
    """
    top_acts = {}
    for feature in features:
        for e in feature["top"][:n]:
            top_acts.setdefault(e["id"], as_doc(e, max_words))["sources"].append(
                {
                    "feature_id": feature["neuron"],
                    "activation_value": float(e["activation_value"]),
                }
            )
    return top_acts


def get_contrast(features, n, max_words, rng):
    """Zero-activating comments for each feature: used to measure precision"""
    docs = {e["id"]: as_doc(e, max_words) for f in features for e in f["neg"]}
    ids = sorted(docs)
    return {i: docs[i] for i in (rng.sample(ids, n) if n and len(ids) > n else ids)}
            


def select_features_for_generation(features, exclude_thread_saturation_threshold):
    """This just removes high-thread-saturation features from the pool of features.
    Return the selected features sorted by descending z-score and the number of dropped features
    """
    pool = [f for f in features if f["max_thread_share"] <= exclude_thread_saturation_threshold]
    return sorted(pool, key=lambda f: -f["z"]), len(features) - len(pool)


def cited_features(facts, features):
    """extract the features only if they are cited in the facts"""
    cited = {fid for fa in facts for fid in fa["feature_ids"]}
    return [f for f in features if int(f["neuron"]) in cited]


def render_feature_block(feature, examples, contrast, max_words):
    header = (f"direction={feature['direction']}  z={feature['z']:.2f}  "
              f"effect={feature['effect']:.4f}  n_activating_docs={feature['n_docs']}  "
              f"max_thread_share={feature['max_thread_share']:.3f}")
    lines = [f"### FEATURE {feature['neuron']}", header,
             "", f"TOP-ACTIVATING COMMENTS ({len(examples)} shown):"]
    lines += [f"[id={e['id']}] {truncate_words(e['body'], max_words)}" for e in examples]
    if contrast:
        lines += ["", ("CONTRAST -- comments where this feature does NOT fire. Never cite "
                       "these; use them to see what is distinctive about the group above, "
                       "and to check your criterion would reject them:")]
        lines += [f"- {truncate_words(e['body'], max_words)}" for e in contrast]
    return "\n".join(lines + [""])


def build_gen_user_prompt(selected_features, args, rng):
    """Build user prompt for generating stylized facts"""
    k = args.n_features
    head = [
        "CORPUS", CORPUS_BLURB, "",
        "TARGET VARIABLE", TARGET_BLURBS.get(args.target, ""), "",
        (f"You are shown {len(selected_features)} candidate features. Each stylized fact may be "
         f"attributed to AT MOST {k} of them -- pick whichever <={k} actually support it, "
         f"not a fixed subset. Different facts should draw on different features where the "
         f"evidence supports it; don't let all {args.n_facts} facts cite the same handful."),
        "",
    ]
    blocks = [render_feature_block(f, rng.sample(f["top"][:args.n_show_examples], args.n_show_examples),
                                   rng.sample(f["neg"], min(N_CONTRAST, len(f["neg"]))),
                                   args.max_words)
              for f in selected_features]
    tail = f"Write exactly {args.n_facts} distinct stylized facts. Return only the JSON object."
    return "\n".join(head + blocks + [tail])


# Errors that will never succeed on retry. Retrying a 400 just burns backoff.
FATAL = (
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
)


class TruncatedResponseError(RuntimeError):
    pass


def call_model(client, model, system, user, max_tokens, cache_dir):
    key = hashlib.sha256("|".join([model, system, user, str(max_tokens)]).encode()).hexdigest()[:32]
    cpath = os.path.join(cache_dir, key + ".json") if cache_dir else None
    if cpath and os.path.exists(cpath):
        with open(cpath) as f:
            cached = json.load(f)
        if cached.get("stop_reason") == "end_turn":
            return cached["text"]
        raise TruncatedResponseError(                       
            f"cached stop_reason={cached['stop_reason']} at max_tokens={max_tokens}")
    for attempt in range(MAX_RETRIES):
        try:
            r = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                       messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            if r.stop_reason in {"max_tokens", "model_context_window_exceeded"}:
                if cpath:
                    os.makedirs(cache_dir, exist_ok=True)
                    with open(cpath, "w") as f:
                        json.dump({"text": text, "model": model,
                                   "stop_reason": r.stop_reason,
                                   "usage": {"in": r.usage.input_tokens,
                                             "out": r.usage.output_tokens}}, f)
                raise TruncatedResponseError(
                    f"stop_reason={r.stop_reason}, output_tokens={r.usage.output_tokens}, "
                    f"max_tokens={max_tokens}")
            if r.stop_reason != "end_turn":
                raise RuntimeError(f"Unexpected stop_reason: {r.stop_reason}")
            if cpath:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cpath, "w") as f:
                    json.dump({"text": text, "model": model, "stop_reason": r.stop_reason,
                               "usage": {"in": r.usage.input_tokens,
                                         "out": r.usage.output_tokens}}, f)
            return text
        except TruncatedResponseError:
            raise
        except Exception as e:  # noqa: BLE001
            if any(t in str(e) for t in FATAL):
                raise RuntimeError(f"{model} rejected the request (not retryable): {e}")
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"API failed after {MAX_RETRIES} tries: {e}")
            time.sleep(min(2 ** (attempt + 1), 30))


def call_with_budget_retry(client, model, system, user, cache_dir):
    """Generation output length is unpredictable; grow max_tokens until it fits."""
    for budget in GEN_BUDGETS:
        try:
            return call_model(client, model, system, user, budget, cache_dir)
        except TruncatedResponseError as e:
            print(f"  ! truncated: {e}", file=sys.stderr)
    raise RuntimeError(f"Response was still truncated at {GEN_BUDGETS[-1]} tokens")


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i == -1 or j <= i:
            raise
        return json.loads(text[i : j + 1])


def normalize_facts(raw, valid_fids, k, n_expected):
    out = []
    for i, fa in enumerate(raw):
        fid = fa.get("fact_id") or f"f{i + 1}"
        feats = [int(x) for x in fa.get("feature_ids", []) if int(x) in valid_fids][:k]
        if not feats:
             print(f"  ! {fid}: no valid feature_ids; dropping", file=sys.stderr)
             continue
        out.append(
            {
                "fact_id": fid,
                "stylized_fact": fa["stylized_fact"].strip(),
                "evidence_criterion": fa.get("evidence_criterion", "").strip(),
                "feature_ids": sorted(feats),
            }
        )
    if len(out) != n_expected:
        print(
            f"  ! model returned {len(out)} facts, asked for {n_expected}",
            file=sys.stderr,
        )
    return out


def generate_facts(client, args, features, cache_dir, rng, system=GEN_SYSTEM, feedback=""):
    """Generate stylized facts from selected features"""
    rng = random.Random(42)
    user_prompt = build_gen_user_prompt(features, args, rng) + feedback
    if args.verbose:
        print(f"--- PROMPT ({len(user_prompt) // 4} est. tokens) ---\n{user_prompt[:1000]}\n...\n{user_prompt[-1000:]}")
    response = call_with_budget_retry(client, args.gen_model, system, user_prompt, cache_dir)
    if args.verbose:
        print(f"--- RESPONSE ---\n{response}")
    return normalize_facts(parse_json(response).get("facts", []),
                            {int(f["neuron"]) for f in features},
                            args.n_features, args.n_facts)


def assign_labels(client, args, facts, docs, cache_dir, tag):
    """Label stylized facts with supporting examples from the selected features"""
    criteria = "\n".join(
        f'{f["fact_id"]}: {f["stylized_fact"]}\n    CRITERION: {f["evidence_criterion"]}'
        for f in facts)
    ids = sorted(docs)
    batches = [ids[i:i + args.attr_batch_size] for i in range(0, len(ids), args.attr_batch_size)]

    def render(batch):
        lines = ["CLAIMS", criteria, "", "COMMENTS"]
        lines += [f"[comment_id={c}] {docs[c]['prompt_body']}" for c in batch]
        lines += ["", "Return the JSON object with one entry per comment_id above."]
        return "\n".join(lines)
    
    assigned = collections.defaultdict(set)
    valid = {f["fact_id"]: f for f in facts}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(call_model, client, args.attr_model, ATTR_SYSTEM, render(b),
                MAX_TOKENS_ATTR, cache_dir): b for b in batches}
        for fut in cf.as_completed(futs):
            allowed = set(futs[fut])
            try:
                labels = parse_json(fut.result()).get("labels", [])
            except Exception as e:                                        # noqa: BLE001
                print(f"  ! attribute batch failed, skipped: {e}", file=sys.stderr)
                continue
            for row in labels:
                cid = int(row.get("comment_id", -1))
                if cid not in allowed:                     # hallucinated / out-of-batch id
                    continue
                for fid in row.get("supports", []):
                    if fid in valid:
                        assigned[fid].add(cid)
            done += 1
            if done % 10 == 0 or done == len(batches):
                print(f"  attribute {done}/{len(batches)}", flush=True)
    return assigned


def eligible_ids(fact, docs):
    """Comments from at least 1 feature this fact is attributed to"""
    fids = set(fact["feature_ids"])
    return {cid for cid, doc in docs.items() if fids & {s["feature_id"] for s in doc["sources"]}}

 
def assemble(facts, labelling_stage):
    docs, assigned, metrics = labelling_stage["docs"], labelling_stage["assigned"], labelling_stage["metrics"]
    by_neuron = {int(f["neuron"]): f for f in labelling_stage["cited"]}
    rows = []
    for fa in facts:
        fids = set(fa["feature_ids"])
        examples = []
        for cid in sorted(set(assigned.get(fa["fact_id"], set())) & eligible_ids(fa, docs)):
            p = docs[cid]
            src = max((s for s in p["sources"] if s["feature_id"] in fids),
                      key=lambda s: s["activation_value"])
            ex = {"comment_id": cid, "body": p["body"], "created_utc": p["created_utc"],
                  "created_utc_str": p["created_utc_str"],
                  "activation_value": src["activation_value"], "feature_id": src["feature_id"]}
            examples.append(ex)
        rows.append({"fact_id": fa["fact_id"],
                     "stylized_fact": fa["stylized_fact"],
                     "evidence_criterion": fa["evidence_criterion"],
                     "metrics": metrics.get(fa["fact_id"], {}),
                     "supporting_examples": examples,
                     "features": [{"feature_id": n, "n_docs": by_neuron[n]["n_docs"],
                                   "z": by_neuron[n]["z"], "effect": by_neuron[n]["effect"],
                                   "direction": by_neuron[n]["direction"],
                                   "max_thread_share": by_neuron[n]["max_thread_share"]}
                                  for n in sorted(fids) if n in by_neuron]})
    return rows


def measure(facts, results, k):
    """Per-fact support / span / precision.
    precision: rate-normalised, since the top_activations and the contrast set differ in size.
    results: metrics and metadata from generate + label run
    """
    docs, assigned = results["docs"], results["assigned"]
    contrast, contrast_assigned = results["contrast"], results["contrast_assigned"]
    out = {}
    for fa in facts:
        fids = set(fa["feature_ids"])
        eligible = eligible_ids(fa, docs)
        hits = set(assigned.get(fa["fact_id"], set())) & eligible
        per_feature = collections.Counter(
            s["feature_id"] for c in hits for s in docs[c]["sources"] if s["feature_id"] in fids)
        spanned = sum(1 for f in fids if per_feature[f])
        rec = {"n_examples": len(hits),
               "n_eligible": len(eligible),
               "support_rate": round(len(hits) / max(len(eligible), 1), 3),
               "n_features_attributed": len(fids),
               "n_features_spanned": spanned,
               "span_ratio": round(spanned / k, 3),
               "examples_per_feature": {str(f): per_feature[f] for f in sorted(fids)}}
        if contrast:
            # rate-normalised: pool and contrast differ in size
            fp = set(contrast_assigned.get(fa["fact_id"], set()))
            tp_rate = len(hits) / max(len(eligible), 1)
            fp_rate = len(fp) / max(len(contrast), 1)
            rec.update({"n_contrast": len(contrast), "n_false_positives": len(fp),
                        "contrast_match_rate": round(fp_rate, 3),
                        "precision": round(tp_rate / (tp_rate + fp_rate), 3)
                        if tp_rate + fp_rate else 0.0})
        out[fa["fact_id"]] = rec
        out[fa["fact_id"]] = rec
    return out



def run_labelling_stage(client, args, facts, features, cache_dir, rng):
    """Pull cited features from facts, pool top-N comments for each, and label facts with at most K * top-N comments"""
    cited = cited_features(facts, features)
    docs = get_top_activations(cited, args.n_top_comments, args.max_words)
    print(f"{len(cited)} cited features of {len(features)} shown; pool {len(docs)} comments "
          f"(per-fact ceiling {args.n_features}x{args.n_top_comments})")
    labelling_stage = {"cited": cited, "docs": docs, "contrast": {}, "contrast_assigned": {}}
    labelling_stage["assigned"] = assign_labels(client, args, facts, docs, cache_dir, "labelling")
    labelling_stage["contrast"] = get_contrast(cited, args.n_score_contrast, args.max_words, rng)
    print(f"contrast {len(labelling_stage['contrast'])} zero-activating comments")
    labelling_stage["contrast_assigned"] = assign_labels(client, args, facts, 
                                                         labelling_stage["contrast"], cache_dir, "contrast")
    labelling_stage["metrics"] = measure(facts, labelling_stage, args.n_features)
    return labelling_stage


def repair_feedback(facts, labelling_stage, args, rng):
    """Render realised performance as a suffix to the generation prompt."""
    docs, metrics = labelling_stage["docs"], labelling_stage["metrics"]
    lines = ["", "REALISED PERFORMANCE OF YOUR PREVIOUS FACTS", ""]
    for fa in facts:
        m = metrics[fa["fact_id"]]
        lines += [f'{fa["fact_id"]}: {fa["stylized_fact"]}',
                  f'    CRITERION: {fa["evidence_criterion"]}',
                  f'    features {fa["feature_ids"]}, spanned {m["n_features_spanned"]}',
                  (f'    support {m["n_examples"]}/{m["n_eligible"]} '
                   f'({m["support_rate"]}); per-feature {m["examples_per_feature"]}')]
        if "precision" in m:
            lines.append(f'    precision {m["precision"]} -- {m["n_false_positives"]} of '
                         f'{m["n_contrast"]} non-activating comments also matched')
            for c in sorted(labelling_stage["contrast_assigned"].get(fa["fact_id"], set()))[:N_FALSE_POS]:
                lines.append(f'      FALSE POSITIVE: {labelling_stage["contrast"][c]["prompt_body"][:220]}')
        misses = sorted(eligible_ids(fa, docs) - set(labelling_stage["assigned"].get(fa["fact_id"], set())))
        for c in rng.sample(misses, min(N_NEAR_MISS, len(misses))):
            lines.append(f'      NEAR MISS: {docs[c]["prompt_body"][:220]}')
        if m["n_examples"] < args.min_examples:
            lines.append(f'    -> support below {args.min_examples}: decide between '
                         f'over-fitted and correct-but-rare. Do not broaden to a topic.')
        if m.get("precision", 1.0) < args.precision_floor:
            lines.append(f'    -> precision below {args.precision_floor}: narrow it.')
        lines.append("")
    return "\n".join(lines + [f"Return all {len(facts)} facts with their original fact_ids."])


def merge_repaired(old_facts, revised):
    """Keep fact_ids stable: ignore invented ids, keep anything repair dropped."""
    old = {f["fact_id"]: f for f in old_facts}
    out = {}
    for f in revised:
        if f["fact_id"] not in old:
            print(f"  ! repair invented {f['fact_id']}; dropping", file=sys.stderr)
            continue
        if f["stylized_fact"] != old[f["fact_id"]]["stylized_fact"]:
            print(f"  {f['fact_id']} revised")
        out[f["fact_id"]] = f
    for fid, f in old.items():
        if fid not in out:
            print(f"  ! repair omitted {fid}; keeping original", file=sys.stderr)
            out[fid] = f
    return [out[k] for k in sorted(out)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n_score_contrast", type=int, default=200)
    parser.add_argument("--min_examples", type=int, default=10,
                   help="repair advisory threshold, not a filter")
    parser.add_argument("--embeddings_file", type=str, default="data/embeddings.npy")
    parser.add_argument("--dedupe_threshold", type=float, default=0.6)
    parser.add_argument("--precision_floor", type=float, default=0.7)
    parser.add_argument(
        "--target",
        type=str,
        default="gpt5_marker",
        choices=["gpt5_marker", "engage", "ontopic"],
    )
    parser.add_argument("--output_dir", type=str, default="data/facts")
    parser.add_argument(
        "--n_features",
        type=int,
        default=10,
        help="Number of features to restrict per stylized fact. K: 1, 5, 10",
    )
    parser.add_argument(
        "--n_facts", type=int, default=10, help="Number of stylized facts to generate."
    )
    parser.add_argument(
        "--n_top_comments",
        type=int,
        default=100,
        help="Number of top-activating comments to read from each feature.",
    )
    parser.add_argument(
        "--n_show_examples",
        type=int,
        default=25,
        help="Number of examples per feature to provide as a prompt to the LLM for stylized fact generation."
    )
    parser.add_argument(
        "--gen_model", default=GEN_MODEL, choices=[GEN_MODEL, ATTR_MODEL]
    )
    parser.add_argument(
        "--attr_model", default=ATTR_MODEL, choices=[GEN_MODEL, ATTR_MODEL]
    )
    parser.add_argument(
        "--attr_batch_size",
        type=int,
        default=25,
        help="Number of comments to label in each batch.",
    )
    parser.add_argument(
        "--exclude_thread_saturation_threshold",
        type=float,
        default=0.5,
        help="Drop features whose top docs concentrate in one thread above this threshold.",
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=100,
        help="Maximum number of words to include in the prompt.",
    )
    parser.add_argument(
        "--cache_dir", default=".sf_cache", help="Directory to cache API calls."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent API calls to make.",
    )
    return parser.parse_args()


def summarise(facts, metrics, k):
    print(f"\n  {'id':<5}{'examples':>9}{'span':>8}{'prec':>7}")
    for fa in facts:
        m = metrics[fa["fact_id"]]
        print(f"  {fa['fact_id']:<5}{m['n_examples']:>9}"
              f"{m['n_features_spanned']:>5}/{k:<2}"
              f"{m.get('precision', float('nan')):>7.2f}")


def main():
    args = parse_args()

    dotenv.load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    client = Anthropic(api_key=api_key)
    rng = random.Random(args.seed)
    
    # for generation, just filter out high-thread-saturation features 
    # (we only want to restrict to n_features for attribution)
    features, dropped = select_features_for_generation(
        load_features_jsonl(args.features_file), args.exclude_thread_saturation_threshold)
    print(f"{len(features)} candidate features (dropped {dropped} above max_thread_share "
          f"{args.exclude_thread_saturation_threshold}); K={args.n_features} per fact")


    if args.embeddings_file:
        deduped_features = dedupe_features(features, args.embeddings_file, args.n_top_comments, args.dedupe_threshold)
        print(f"deduped {len(features)} features -> {len(deduped_features)} concepts "
              f"(threshold {args.dedupe_threshold})")

    stylized_facts = generate_facts(client, args, deduped_features, args.cache_dir, rng)
    if not stylized_facts:
        raise RuntimeError("no valid facts generated")
    labelling_stage = run_labelling_stage(client, args, stylized_facts, deduped_features, args.cache_dir, rng)

    # print(f"repairing {len(stylized_facts)} facts...")
    # revised = generate_facts(client, args, deduped_features, args.cache_dir, rng,
    #                              system=REPAIR_SYSTEM,
    #                              feedback=repair_feedback(stylized_facts, labelling_stage, args, rng))
    # stylized_facts = merge_repaired(stylized_facts, revised)
    # labelling_stage = run_labelling_stage(client, args, stylized_facts, deduped_features, args.cache_dir, rng)

    out_path = os.path.join(args.output_dir, f"stylized_facts_{args.target}_{args.n_features}k.jsonl")  
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in assemble(stylized_facts, labelling_stage))
    summarise(stylized_facts, labelling_stage["metrics"], args.n_features)
    print(f"\nwrote {len(stylized_facts)} facts -> {out_path}")

if __name__ == "__main__":
    main()
