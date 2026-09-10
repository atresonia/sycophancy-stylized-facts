# CLAUDE.md

Operational notes for working in this repo. See `README.md` for the research methodology and
motivation; this file is about which script does what, in what order, and what to watch for.

## What this repo does

Studies how r/ChatGPT commenters talk about chatbot sycophancy, via SAE features/neurons run over
comment embeddings. The active line of work is the **per-neuron stylized-facts pipeline**: a
human reads a neuron's top-activating comments, writes "stylized facts" (Hirschman 2016 sense —
simple, existence-only empirical regularities, not quantified claims) by hand, and those facts get
mechanically checked against the neuron's full activating pool by an LLM.

## Pipeline, in order

1. **Sample a neuron's pool** — `src/sample_neuron_pool.py --neuron <id>`. Builds the full
   top-decile pool of that neuron's activating comments plus a keyword/random negative pool,
   caches it to `data/pools_<id>.json`, and prints/optionally saves a sample (default 30 positive
   + 30 negative) for a human to read. `--pos_mode` controls how the positive sample is drawn:
   `random` (default, a spread across the top-decile pool) or `top` (the single highest-activation
   docs). The negative sample is always random — every negative-pool doc has activation == 0 by
   construction, so there's no per-doc score to rank by.
2. **Write manual stylized facts** — fully manual, stays that way. The researcher reads the
   sample from step 1 and writes facts with cited `doc_id`s, saved as
   `data/<neuron>/<neuron>_manual_stylized_facts.txt`. Do not auto-generate or overwrite this file.
3. **Rewrite into candidate facts** — a Claude Code chat rewrites the manual `.txt` into
   `data/<neuron>/candidate_facts.json` (`{fact_id, stylized_fact, evidence_criterion,
   seed_examples}`), following `src/prompt_templates/candidate_fact_rewrite.md` verbatim. That
   template's Rule 1 exists because of a real bug: a manual fact about wanting naturalness got
   rewritten into a fact about the model being unnatural — same topic, inverted claim. Check for
   that failure mode specifically when reviewing a rewrite.
4. **Label + verify** — `src/label_facts.py --neuron <id> --facts_file data/<id>/candidate_facts.json`.
   Batch-labels every fact against the neuron's pool, majority-votes across `--repeats` (default
   2) independent passes, then individually re-verifies each batch "yes" (one doc per call,
   quote-grounded) since batching lets a hit bleed onto a topically-adjacent neighbor. Writes
   `data/<neuron>/labelled_facts.json` — **every fact, pass or fail**, plus `demoted_examples` for
   anything verification knocked down, plus an end-of-run recall check against each fact's
   `seed_examples`.
5. **Roll up across neurons** — `src/generate_rollups.py`. Regenerates
   `data/stylized_facts_consolidated.md` (short, passing facts + a "not yet passing" review
   section) and `data/stylized_facts_evidence.md` (full untruncated quotes) from every
   `data/*/labelled_facts.json` on disk. Safe to re-run any time; fully derived, no hand-editing.

## Conventions that matter

- **Existence restriction on stylized facts.** A fact must be verifiable by reading the cited
  comments alone — no quantifiers (most/many/few/majority/increasingly/...), no causal language
  ("because", "led to"), no prevalence claims. "Commenters do X", not "Some commenters do X".
- **`evidence_criterion` is one operational sentence** a second reader (or a model with no other
  context) could mechanically apply, following the existing `"...Does not count if..."` exclusion
  pattern already used throughout `candidate_facts.json` files.
- **The regex test**: if a criterion could be implemented as a keyword match, it's a vocabulary
  observation, not a stylized fact. State what commenters claim/argue, not which words they use.
- `label_facts.py`'s floor (`max(floor_min, ceil(floor_rate * n_docs))`) requires a fact to hold in
  roughly 10% of the neuron's top-decile pool. Failing the floor is not evaluated as "wrong" —
  `generate_rollups.py`'s "not yet passing" section exists precisely so a fact with real but
  below-floor evidence stays visible instead of silently disappearing.

## Known gotcha: numpy operator precedence

`&` binds tighter than comparisons in Python, so `arr > 0 & mask` parses as `arr > (0 & mask)`,
silently zeroing out the mask instead of applying it. This exact bug existed in the old
`load_act_pools.ipynb` (now replaced by `sample_neuron_pool.py`, which uses `(arr > 0) & mask`) and
inflated a neuron's top-decile size by ~4-5% by including a few unclean docs in the pool-size
denominator. Always parenthesize comparisons before combining them with `&`/`|`.

## Environment

- `ANTHROPIC_API_KEY` via `src/.env` (see `src/.env.example`). `label_facts.py` runs real,
  billed API calls — `--repeats` and `--concurrency` both scale call volume, so check
  `--facts_file`'s fact count before a full run.
- Corpus: `data/r_chatGPT_comments_relating_sycophan_2022-12-22_2026-8-7.jsonl` (20,560 raw
  comments, uncleaned — cleaning happens on the activations via each script's own `clean_data`,
  not on this file) + `data/comments_relating_sycophan_gemini_d131072_sparse_activations.npz`
  (matching sparse SAE activations, same row order).

## Targeted feature selection (not part of the main pipeline)

`select_features.py` (and its cached output, `data/features/*.jsonl`) selects SAE features
against a target (`gpt5_marker`/`engage`/`ontopic`, see `README.md`). It isn't a step in the
neuron-level pipeline above, which starts from a single neuron already chosen — kept available
for whenever a new targeted selection is needed.
