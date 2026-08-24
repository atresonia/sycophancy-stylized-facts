"""Script to run a targeted feature selection on the sparse activations from our general purpose SAE.

Outputs:
    jsonl file, one line per selected feature:
        neuron: feature index of the SAE
        n_docs: number of activating documents post-clean
        z: effect / standard error
        effect: difference in activation means for positive and negative targets
        direction: sign of z
        max_thread_share: largest fraction of a feature's activating docs coming from a single thread
        top / mid / neg: example activating documents for LLM interpretation

Usage:
    python select_features.py --target gpt5_marker --output top_features_gpt5.jsonl
    python select_features.py --target engage --output top_features_engage.jsonl
    python select_features.py --target ontopic --output top_features_ontopic.jsonl
"""

import argparse
import collections
import os
import json
import re
import scipy.sparse as sp
import numpy as np
import datetime

MIN_CHAR = 40
MIN_DENSITY = 100
MAX_DENSITY = 5000
MIN_THREAD_SIZE = 5 # min comments in a thread (for engagement target)

N_FEATURES = 30
N_TOP, N_MID, N_NEG = 100, 70, 50

GPT5_RELEASE = datetime.datetime(2025, 8, 7, tzinfo=datetime.timezone.utc)


def to_py(o):
    """json default hook: unwrap numpy scalars."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')


def load_npz(file_path):
    """Load npz sparse activation file into a dictionary."""
    data = np.load(file_path)
    N, F = int(data['shape'][0]), int(data['shape'][1])
    return sp.csr_matrix((data['values'].astype(np.float32),
                       (data['row_indices'], data['feature_indices'])),
                        shape=(N, F)).tocsc()


def load_jsonl(file_path):
    """Load jsonl file into a list of dictionaries."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def clean_data(data, min_char=MIN_CHAR):
    """Remove automated bot, deleted/removed, and comments less than MIN_CHAR characters."""
    bad_authors = {'AutoModerator', 'ChatGPT-ModTeam'}
    seen = set()
    m = np.zeros(len(data), dtype=bool)
    for i, row in enumerate(data):
        body = row.get('body', '').strip()
        ok = (row.get('author') not in bad_authors
              and len(body) >= min_char
              and body not in ('[removed]', '[deleted]', '')
              and body not in seen)
        m[i] = ok
        if ok:
            seen.add(body)
    return m


def build_target(data, target, keep):
    """Build the target vector
       data: (num_comments, ): list of comment dictionaries (ex size: 20560)
       target: 'gpt5_marker', 'engage', 'ontopic'
       keep: (num_comments, ): boolean vector of comments to keep after cleaning
       Outputs: (num_comments, ): target vector: 1 if comment is relevant to the target, 0 otherwise
    """
    ts = np.array([r['created_utc'] for r in data])
    sc = np.array([r.get('score', 0) for r in data])
    out = np.full(len(data), np.nan)

    if target == 'gpt5_marker':
        out[keep] = (ts[keep] >= GPT5_RELEASE.timestamp()).astype(float)
        return out
    elif target == 'engage':
        y = np.log1p(np.clip(sc, 0, None))
        groups = collections.defaultdict(list)
        for i, row in enumerate(data):
            if keep[i]:
                groups[row['link_id']].append(i)
        for k, ix in groups.items():
            if len(ix) < MIN_THREAD_SIZE:
                continue
            a = y[ix]
            if a.std() < 1e-6:
                continue
            out[ix] = (a - a.mean()) / a.std()
        return out
    elif target == 'ontopic':
        pat = re.compile(r'sycophan|glaz|flatter|yes.?man|kiss.?ass|ass.?kiss|'
                         r'brown.?nos|boot.?lick|pander|suck.?up', re.I)
        out[keep] = np.array([1.0 if pat.search(r['body']) else 0.0 for r in data])[keep]
        return out
    raise ValueError(f"Invalid target: {target}")


def score_binary_features(P, y):
    """P: binary feature matrix (T if activation is non-zero, F otherwise) (17405 x 4182)
       y: 1) gpt5_marker: binary target indicating if comment was gpt5 and after (1) or before (0)
          2) ontopic: binary target indicating if comment is sycophancy-related (1) or not (0)
        Outputs: 1) z-score: p1 - p0 / se: confidence of the effect size (how many standard deviations the difference is from 0)
                 2) effect-size: p1 - p0 (difference between activation means for positive and negative targets)
    """
    p1 = P[y == 1].mean(0)
    p0 = P[y == 0].mean(0)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    effect = p1 - p0
    return effect / se, effect


def score_continous_features(P, y):
    """P: binary feature matrix (T if activation is non-zero, F otherwise) (17405 x 4182)
        y: 1) engagement: continuous target indicating the log score of the comment
        Outputs: 1) z-score: p1 - p0 / se: confidence of the effect size (how many standard deviations the difference is from 0)
                 2) effect-size: mean activation for positive target - mean activation for negative target
    """
    n = P.sum(0).astype(float)
    s1 = (P * y[:, None]).sum(0)
    m1 = np.where(n > 0, s1 / np.maximum(n, 1), 0)
    m0 = (y.sum() - s1) / np.maximum(len(y) - n, 1)
    var = y.var()
    se = np.sqrt(var / np.maximum(n, 1) + var / np.maximum(len(y) - n, 1)) + 1e-9
    effect = m1 - m0
    return effect / se, effect


def thread_share(link_ids, doc_idx):
    """Largest fraction of a feature's activating docs coming from a single thread."""
    if len(doc_idx) == 0:
        return 0.0
    c = collections.Counter(link_ids[j] for j in doc_idx)
    return max(c.values()) / len(doc_idx)


def make_records(idx, vals, body, data):
    out = []
    for j, d in enumerate(idx):
        d = int(d)
        ts = data[d]['created_utc']
        out.append({
            'activation_value': float(vals[j]),
            'id': d,
            'body': body[d],
            'created_utc': ts,
            'created_utc_str': datetime.datetime.fromtimestamp(
                ts, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        })
    return out


def main():
    data_dir = "data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, 
                        default="r_chatGPT_comments_relating_sycophan_2022-12-22_2026-8-7.jsonl", 
                        help="input JSONL file name. Expected to be in the data folder.")
    parser.add_argument("--input_npz", type=str, 
                        default="comments_relating_sycophan_gemini_d131072_sparse_activations.npz", 
                        help="input NPZ file name. Expected to be in the data folder.")
    parser.add_argument("--target", type=str, required=True, help="options: gpt5_marker, engage, ontopic")
    parser.add_argument("--output", type=str, required=True, 
                        help="output JSONL file name. Will be saved to data/results/.")
    args = parser.parse_args()
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    output_file = os.path.join(results_dir, args.output)
    print(f"Selecting features for {args.target} and outputting to {output_file}")

    data = load_jsonl(os.path.join(data_dir, args.input_jsonl))
    X = load_npz(os.path.join(data_dir, args.input_npz))
    body = [r.get('body', '') for r in data]
    link_ids = [r.get('link_id') for r in data]
    
    keep = clean_data(data)
    y = build_target(data, args.target, keep)
    
    clean_idx = np.flatnonzero(keep & np.isfinite(y))
    yc = y[clean_idx]
    print(f"{len(clean_idx)}/{len(y)} comments kept after cleaning")
    if args.target == 'engage':
        print(f"  target: mean={yc.mean():.3f} sd={yc.std():.3f} "
              f"min={yc.min():.2f} max={yc.max():.2f}")
    else:
        print(f"  target: base rate={yc.mean():.3f} "
              f"n1={int((yc == 1).sum())} n0={int((yc == 0).sum())}")

    Xc = X[clean_idx]
    act_cnts = np.asarray((Xc > 0).sum(0)).ravel() # number of non-zero activations per feature
    keep_feats = np.flatnonzero((act_cnts >= MIN_DENSITY) & (act_cnts <= MAX_DENSITY))
    if keep_feats.size == 0:
        raise RuntimeError("No features passed the density filter.")
    P = (Xc[:, keep_feats] > 0).toarray()
    print(f"  {keep_feats.size} features after density filter")

    if args.target == 'gpt5_marker' or args.target == 'ontopic':
        z, eff = score_binary_features(P, yc)
    elif args.target == 'engage':
        z, eff = score_continous_features(P, yc)
    else:
        raise ValueError(f"Invalid target: {args.target}")
    
    n_docs = P.sum(0)
    valid = np.flatnonzero(np.isfinite(z))
    if valid.size == 0:
        raise RuntimeError("No features passed the validity filter.")
    zv = z[valid]
    pos = valid[np.argsort(-zv)][:N_FEATURES]
    negd = valid[np.argsort(zv)][:N_FEATURES]
    order = list(dict.fromkeys(np.concatenate([pos, negd]).tolist()))
    print(f"  writing {len(order)} features")
    
    rng = np.random.default_rng(0)
    # write to jsonl file {neuron: , n_docs: , z: , effect: , direction: , max_thread_share: , top: [{activation_value:, id: , body: , created_utc: }], mid: [..], neg: [..]}
    with open(output_file, 'w') as f:
        for i in order:
            fi = int(keep_feats[i])
            col = np.asarray(X[:, fi].todense()).ravel()
            hit = clean_idx[P[:, i]]
            hv = col[hit]
            o = np.argsort(-hv)
            ti = o[:N_TOP]
            mi = o[N_TOP:N_TOP + N_MID]

            pool = clean_idx[~P[:, i]]
            k = min(N_NEG, len(pool))
            neg = rng.choice(pool, k, replace=False) if k else np.array([], dtype=int)
            
            f.write(json.dumps({
                'neuron': fi,
                'n_docs': len(hit),
                'z': z[i],
                'effect': eff[i],
                'direction': 'positive' if z[i] > 0 else 'negative',
                'max_thread_share': thread_share(link_ids, hit),
                'top': make_records(hit[ti], hv[ti], body, data),
                'mid': make_records(hit[mi], hv[mi], body, data),
                'neg': make_records(neg, np.zeros(len(neg)), body, data),
            }, default=to_py) + '\n')

if __name__ == "__main__":

    main()