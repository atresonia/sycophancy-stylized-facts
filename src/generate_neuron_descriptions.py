"""Script to generate descriptions for a given neuron."""

import asyncio, json, math, os, random, re, collections
import numpy as np
import scipy.sparse as sp
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import csv
import itertools
import sys

load_dotenv()

MODEL = "claude-sonnet-4-5"
NEURON = 39774
NPZ = "data/comments_relating_sycophan_gemini_d131072_sparse_activations.npz"
JSONL = "data/r_chatGPT_comments_relating_sycophan_2022-12-22_2026-8-7.jsonl"
DATA_DIR = f"data/{NEURON}"
N_LABEL = 80
BATCH, CONCURRENCY = 30, 8

COMPONENTS = {
    "P": "The author describes something they personally experienced with an AI chatbot (not a general claim about AI).",
    "B": "The author names a specific concrete behavior of the model (a phrasing, a follow-up question, a tone, a refusal).",
    "F": "The author mentions a fix they tried: a prompt, custom instruction, setting, or workaround.",
    "C": "The author frames it as a recent change in the model's behavior.",
}

# ---------------------------------------------------------------- data
def load_npz(path):
    d = np.load(path)
    N, F = int(d['shape'][0]), int(d['shape'][1])
    return sp.csr_matrix((d['values'].astype(np.float32),
                          (d['row_indices'], d['feature_indices'])),
                         shape=(N, F)).tocsc()

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]
 
 
def load_all():
    matrix, data = load_npz(NPZ), load_jsonl(JSONL)
    a = matrix[:, NEURON].toarray().ravel()
    clean = clean_data(data)
    return data, a, clean


def clean_data(data, min_char=40):
    bad = {'AutoModerator', 'ChatGPT-ModTeam'}
    seen, m = set(), np.zeros(len(data), dtype=bool)
    for i, row in enumerate(data):
        body = row.get('body', '').strip()
        ok = (row.get('author') not in bad and len(body) >= min_char
              and body not in ('[removed]', '[deleted]', '') and body not in seen)
        m[i] = ok
        if ok:
            seen.add(body)
    return m


def build_pools(data, activations, clean, n_neg=120, seed=42):
    rng = np.random.default_rng(seed)
    active = np.flatnonzero((activations > 0) & clean)
    inactive = np.flatnonzero((activations == 0) & clean)
    active = active[np.argsort(-activations[active])]
 
    top_dec = active[:max(1, len(active) // 10)]
    kw = re.compile(r'sycophan', re.I)
    inact_kw = np.array([i for i in inactive if kw.search(data[i]['body'])], dtype=int)
    n_kw = min(n_neg // 2, len(inact_kw))
    neg_kw = rng.choice(inact_kw, n_kw, replace=False) if n_kw else np.array([], dtype=int)
    rest = np.setdiff1d(inactive, neg_kw)
    neg_rand = rng.choice(rest, min(n_neg - n_kw, len(rest)), replace=False)
 
    rows = lambda idx: [{'doc_id': data[i]['id'], 'text': data[i]['body'], 'row': int(i)}
                        for i in idx]
 
    pools = {'pos_dec': rows(top_dec),
             'neg_kw': rows(neg_kw), 'neg_rand': rows(neg_rand)}
    print({k: len(v) for k, v in pools.items()})
    return pools


# ---------------------------------------------------- log-odds contrast
 
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being it its it's this that these those i i'm i've my me we our you your they them their he she
his her do don't don doesn did not no so just what which who whom how when where why can could
will would should have has had get got go going one two few last end o s t re ve ll m d""".split())
 
 
def contrast(pos, neg, min_df=12, top=20, bottom=10):
    """Smoothed log odds ratio per word. +0.5 is the Haldane-Anscombe correction,
    which keeps the ratio finite when a count is zero
    (Haldane 1956, Annals of Human Genetics 20(4):309-311)."""
    tok = lambda t: set(re.findall(r"[a-z']+", t.lower()))
    cp, cn = collections.Counter(), collections.Counter()
    for t in pos: cp.update(tok(t))
    for t in neg: cn.update(tok(t))
    NP, NN, rows = len(pos), len(neg), []
    for w in set(cp) | set(cn):
        a, b = cp[w], cn[w]
        if a + b < min_df or w in STOP or len(w) <= 2:
            continue
        pr, nr = a / NP, b / NN
        if pr > 0.5:                       # present in most positives -> no discriminative use
            continue
        lo = math.log(((a + .5) / (NP - a + .5)) / ((b + .5) / (NN - b + .5)))
        rows.append({'word': w, 'pos': round(pr, 2), 'neg': round(nr, 2), 'lo': lo})
    rows.sort(key=lambda r: -r['lo'])
    strip = lambda rs: [{k: r[k] for k in ('word', 'pos', 'neg')} for r in rs]
    return strip(rows[:top]), strip(rows[-bottom:])


# ---------------------------------------------------------- generation
FRAMINGS = [
    "Name the property that separates the two groups. Be specific enough to be wrong.",
    "What is Group A about that Group B is not about?",
    "State the rule the feature applies, as a predicate over documents.",
    "Describe the stance or situation the Group A authors are in that Group B authors are not.",
    "What is happening to the Group A authors that is not happening to Group B authors?",
]

PROMPT = """{framing}
 
GROUP A (feature fires):
{pos}
 
GROUP B (does not fire):
{neg}
 
Words more frequent in A: {hi}
Words more frequent in B: {lo}

Write one sentence, 25 words max, describing what a Group A comment is about or
what its author is doing. Not which words, pronouns, or punctuation it contains.
It must be narrow enough to exclude most of Group B: if your sentence would apply
to most comments in a ChatGPT subreddit, it is too broad. Output the sentence only."""

BANNED = ('often', 'usually', 'tends to', 'may ', 'e.g.', 'such as',
          'rather than', 'typically', 'sometimes', 'generally')
SURFACE = ('word "', "word '", 'pronoun', 'punctuation', 'apostrophe',
           'contraction', 'quotation mark', 'question mark', 'contains the word')


def acceptable(s):
    low = s.lower()
    if len(s.split()) > 25: return False, 'too long'
    if low.count(',') + low.count(' or ') + low.count(' and ') > 2:
        return False, 'list of alternatives'
    for b in BANNED:
        if b in low: return False, f'hedge: {b.strip()}'
    for b in SURFACE:
        if b in low: return False, f'surface: {b.strip()}'
    return True, ''


async def sample_candidates(pools, k=10, tries=3, seed=0):
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rng = np.random.default_rng(seed)
    hi, lo = contrast([d['text'] for d in pools['pos_dec']],
                      [d['text'] for d in pools['neg_kw'] + pools['neg_rand']])
 
    async def one(j):
        for _ in range(tries):
            pos = list(rng.choice(pools['pos_dec'], rng.integers(10, 21), replace=False))
            neg = list(rng.choice(pools['neg_kw'], 12, replace=False)) + \
                  list(rng.choice(pools['neg_rand'], 6, replace=False))
            body = PROMPT.format(
                framing=FRAMINGS[j % len(FRAMINGS)],
                pos=json.dumps([d['text'] for d in pos]),
                neg=json.dumps([d['text'] for d in neg]),
                hi=json.dumps(hi), lo=json.dumps(lo)
            )
            r = await client.messages.create(model=MODEL, max_tokens=200, temperature=1.0,
                                             messages=[{"role": "user", "content": body}])
            s = r.content[0].text.strip().strip('"')
            ok, why = acceptable(s)
            if ok:
                return s
            print(f"  rejected ({why}): {s[:70]}")
        return None
 
    out = [c for c in await asyncio.gather(*[one(j) for j in range(k)]) if c]
    print(f"{len(out)}/{k} candidates survived")
    return out


# ------------------------------------------------------- labeling sheet
 
def make_label_sheet(data, a, clean, seed=11):
    """Blind, stratified. Actives are sampled across the WHOLE active range, not
    just the top decile the candidates were written from -- otherwise you only
    learn that the description fits the examples it was generated from.
    Half the inactives contain the keyword, so 'mentions sycophancy' cannot win."""
    rng = np.random.default_rng(seed)
    act = np.flatnonzero((a > 0) & clean)
    inact = np.flatnonzero((a == 0) & clean)
    act = act[np.argsort(-a[act])]
 
    n_a = N_LABEL // 2
    thirds = np.array_split(act, 3)                      # strong / middle / weak
    pick_a = np.concatenate([rng.choice(t, n_a // 3, replace=False) for t in thirds])
 
    kw = re.compile(r'sycophan', re.I)
    ikw = np.array([i for i in inact if kw.search(data[i]['body'])], dtype=int)
    irest = np.setdiff1d(inact, ikw)
    n_i = N_LABEL - len(pick_a)
    pick_i = np.concatenate([rng.choice(ikw, n_i // 2, replace=False),
                             rng.choice(irest, n_i - n_i // 2, replace=False)])
 
    idx = np.concatenate([pick_a, pick_i])
    rng.shuffle(idx)                                     # blind: no order signal
 
    os.makedirs(DATA_DIR, exist_ok=True)
    json.dump({str(int(i)): bool(a[i] > 0) for i in idx},
              open(f'{DATA_DIR}/label_key.json', 'w'))       # key kept OUT of the sheet
    with open(f'{DATA_DIR}/label_sheet.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row', 'text'] + list(COMPONENTS))
        for i in idx:
            w.writerow([int(i), data[i]['body'].replace('\n', ' ')] + [''] * len(COMPONENTS))
    print(f"wrote {DATA_DIR}/label_sheet.csv ({len(idx)} docs)")
    for k, v in COMPONENTS.items():
        print(f"  {k}: {v}")
    print("\nFill each component column with 1 or 0. Do not open label_key.json.")


# ------------------------------------------------------------- analysis
 
def wilson(k, n, z=1.96):
    """Wilson score interval -- better than the normal approximation at small n
    (Wilson 1927, JASA 22:209-212)."""
    if n == 0: return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)
 
 
def mcc(y, yh):
    tp = int(((y == 1) & (yh == 1)).sum()); tn = int(((y == 0) & (yh == 0)).sum())
    fp = int(((y == 0) & (yh == 1)).sum()); fn = int(((y == 1) & (yh == 0)).sum())
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else (tp * tn - fp * fn) / den
 
 
def boot_mcc(y, yh, n=2000, seed=3):
    rng = np.random.default_rng(seed)
    vals = [mcc(y[s], yh[s]) for s in
            (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])
 
 
def evaluate(name, y, yh):
    tp = int(((y == 1) & (yh == 1)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum())
    m = mcc(y, yh); ci = boot_mcc(y, yh)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    pl, ph = wilson(tp, max(1, tp + fp)); rl, rh = wilson(tp, max(1, tp + fn))
    flag = ''
    if yh.mean() > 0.9: flag = '  VACUOUS'
    elif ci[0] <= 0: flag = '  not distinguishable from chance'
    return {'name': name, 'mcc': m, 'ci': ci, 'precision': prec, 'recall': rec,
            'p_ci': (pl, ph), 'r_ci': (rl, rh), 'pos_rate': yh.mean(), 'flag': flag}
 
 
def analyze():
    key = {int(k): v for k, v in json.load(open(f'{DATA_DIR}/label_key.json')).items()}
    rows = list(csv.DictReader(open(f'{DATA_DIR}/label_sheet.csv')))
    rows = [r for r in rows if all(r[c].strip() in ('0', '1') for c in COMPONENTS)]
    if len(rows) < N_LABEL:
        print(f"warning: {len(rows)}/{N_LABEL} rows labeled\n")
    y = np.array([key[int(r['row'])] for r in rows], int)
    L = {c: np.array([int(r[c]) for r in rows]) for c in COMPONENTS}
    print(f"labeled {len(y)} docs, {y.mean():.0%} active\n")
 
    results = [evaluate(c, y, L[c]) for c in COMPONENTS]
    for a, b in itertools.combinations(COMPONENTS, 2):
        results.append(evaluate(f"{a} AND {b}", y, L[a] & L[b]))
        results.append(evaluate(f"{a} OR {b}", y, L[a] | L[b]))
    for a, b, c in itertools.combinations(COMPONENTS, 3):
        results.append(evaluate(f"{a} AND {b} AND {c}", y, L[a] & L[b] & L[c]))
 
    results.sort(key=lambda r: -r['mcc'])
    print(f"{'formula':22s} {'MCC':>6s} {'95% CI':>16s}  {'prec':>12s} {'recall':>12s}  rate")
    for r in results:
        print(f"{r['name']:22s} {r['mcc']:+.3f} "
              f"[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]  "
              f"{r['precision']:.2f}[{r['p_ci'][0]:.2f},{r['p_ci'][1]:.2f}] "
              f"{r['recall']:.2f}[{r['r_ci'][0]:.2f},{r['r_ci'][1]:.2f}]  "
              f"{r['pos_rate']:.2f}{r['flag']}")
 
    best = results[0]
    print(f"\nbest formula: {best['name']}")
    print("components in it:")
    for c in re.findall(r'\b[A-Z]\b', best['name']):
        print(f"  {c}: {COMPONENTS[c]}")
    print("\nWrite the final description as one sentence expressing exactly this "
          "formula, then check it against the candidates in candidates.json.")
    return results


# ---------------------------------------------------------------- main
 
if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'generate'
    data, a, clean = load_all()
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"neuron {NEURON}: {int(((a>0)&clean).sum())} active / {int(clean.sum())} clean docs\n")
 
    if cmd == 'generate':
        pools = build_pools(data, a, clean)
        cands = asyncio.run(sample_candidates(pools, k=10))
        json.dump(cands, open(f'{DATA_DIR}/candidates.json', 'w'), indent=1)
        for i, c in enumerate(cands):
            print(f"{i}. {c}")
        print("\nRead these, then edit COMPONENTS at the top of this file so the "
              "components cover them. Then: python describe.py label")
 
    elif cmd == 'label':
        make_label_sheet(data, a, clean)
 
    elif cmd == 'analyze':
        analyze()
  
    else:
        print(__doc__)
 



