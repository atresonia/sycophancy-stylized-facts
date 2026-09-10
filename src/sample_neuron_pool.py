"""Sample a neuron's top-decile activating comments, plus a keyword/random negative pool, for
the manual stylized-fact-writing step (stage 1 of the pipeline -- see CLAUDE.md).

Args:
    --neuron: The index of the neuron to sample from.
    --n_pos: The number of top-decile docs to sample for manual reading. (default: 30)
    --pos_mode: sample mode for the top-decile pool ('random': randomly sample from top-decile (default), 
                'top': the highest-activation docs from top-decile)
    --n_neg: The number of negative/contrast docs to sample for manual reading. (default: 30)
    --pool_neg_size: The size of the full negative pool (half keyword-matched, half random) that --n_neg is drawn from. (default: 120)
    --seed: The random seed to use for sampling. (default: 42)
    --pools_out: The path to cache the full top-decile/negative pools. (default: data/pools_<neuron>.json)
    --sample_out: The path to save the drawn sample as JSON. (default: print only)

Usage:
    python src/sample_neuron_pool.py --neuron 83812
    python src/sample_neuron_pool.py --neuron 83812 --pos_mode top --n_pos 10 --n_neg 10 --pool_neg_size 100 --seed 42 --pools_out data/pools_83812.json --sample_out data/sample_83812.json
"""
import argparse
import json
import re

import numpy as np
import scipy.sparse as sp

NPZ = "data/comments_relating_sycophan_gemini_d131072_sparse_activations.npz"
JSONL = "data/r_chatGPT_comments_relating_sycophan_2022-12-22_2026-8-7.jsonl"


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


def build_pools(matrix, data, neuron_idx, n_neg, seed):
    """Full top-decile pool (uncapped, unlike label_facts.py's cost-bounded version meant for
    LLM labelling -- this is for a human to read) plus a keyword/random negative pool."""
    rng = np.random.default_rng(seed)
    activations = matrix[:, neuron_idx].toarray().ravel()
    clean = clean_data(data)
    active = np.flatnonzero((activations > 0) & clean)
    inactive = np.flatnonzero((activations == 0) & clean)
    active = active[np.argsort(-activations[active])]

    top_dec = active[: max(1, len(active) // 10)]
    kw = re.compile(r"sycophan", re.I)
    inactive_kw = np.array([i for i in inactive if kw.search(data[i]["body"])], dtype=int)
    n_kw = min(n_neg // 2, len(inactive_kw))
    neg_kw = rng.choice(inactive_kw, n_kw, replace=False) if n_kw else np.array([], dtype=int)
    rest = np.setdiff1d(inactive, neg_kw)
    neg_rand = rng.choice(rest, min(n_neg - n_kw, len(rest)), replace=False)

    def rows(idx, with_rank):
        out = []
        for r, i in enumerate(idx):
            row = {"doc_id": data[i]["id"], "text": data[i]["body"],
                   "created_utc": data[i].get("created_utc"), "activation": float(activations[i])}
            if with_rank:
                row["rank"] = r + 1
            out.append(row)
        return out

    return {"pos_dec": rows(top_dec, True), "neg_kw": rows(neg_kw, False), "neg_rand": rows(neg_rand, False)}


def safe_sample(rng, pool, n):
    """Sample without replacement -- n docs, or the full pool if n exceeds its size."""
    if not pool:
        return []
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


def top_n(pool, n):
    """First n docs of a pool already sorted descending by activation (pos_dec carries 'rank')."""
    return pool[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neuron", type=int, required=True)
    ap.add_argument("--n_pos", type=int, default=30,
                     help="Top-decile docs to sample for manual reading.")
    ap.add_argument("--pos_mode", choices=["random", "top"], default="random",
                     help="'random': uniform sample from the top-decile pool (default). "
                          "'top': the --n_pos highest-activation docs (rank 1..n_pos).")
    ap.add_argument("--n_neg", type=int, default=30,
                     help="Negative/contrast docs to sample for manual reading.")
    ap.add_argument("--pool_neg_size", type=int, default=120,
                     help="Size of the full negative pool (half keyword-matched, half random) "
                          "that --n_neg is drawn from.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pools_out", type=str, default=None,
                     help="Where to cache the full top-decile/negative pools "
                          "(default: data/pools_<neuron>.json).")
    ap.add_argument("--sample_out", type=str, default=None,
                     help="Optional path to also save the drawn sample as JSON (default: print only).")
    args = ap.parse_args()

    matrix = load_npz(NPZ)
    data = load_jsonl(JSONL)
    pools = build_pools(matrix, data, args.neuron, args.pool_neg_size, args.seed)

    pools_path = args.pools_out or f"data/pools_{args.neuron}.json"
    with open(pools_path, "w") as f:
        json.dump(pools, f, indent=4)
    print({k: len(v) for k, v in pools.items()})
    print(f"wrote full pool cache to {pools_path}\n")

    rng = np.random.default_rng(args.seed)
    pos_sample = (top_n(pools["pos_dec"], args.n_pos) if args.pos_mode == "top"
                  else safe_sample(rng, pools["pos_dec"], args.n_pos))
    neg_combined = pools["neg_kw"] + pools["neg_rand"]
    neg_sample = safe_sample(rng, neg_combined, args.n_neg)

    print(f"--- top-decile sample, pos_mode={args.pos_mode} ({len(pos_sample)} docs) ---")
    print(json.dumps(pos_sample, indent=2))
    print(f"\n--- negative/contrast sample ({len(neg_sample)} docs) ---")
    print(json.dumps(neg_sample, indent=2))

    if args.sample_out:
        with open(args.sample_out, "w") as f:
            json.dump({"pos_sample": pos_sample, "neg_sample": neg_sample}, f, indent=2)
        print(f"\nwrote sample to {args.sample_out}")


if __name__ == "__main__":
    main()
