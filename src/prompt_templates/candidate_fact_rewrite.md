# Candidate fact rewrite — prompt template

## v1 — 2026-09-10

Paste this whole file's **PROMPT** section into a Claude Code chat, followed by the neuron's
manual writeup (`data/<neuron>/<neuron>_manual_stylized_facts.txt`), each time you rewrite that
neuron's manually-written stylized facts into `data/<neuron>/candidate_facts.json`. This step
stays a human-in-the-loop chat, not a script — this file exists so the instructions are the same
every time, instead of reconstructed from memory, and so failures have one place to get fixed.

Why this exists: rewriting facts by improvised chat produced at least one confirmed meaning-drift
bug (see Rule 1) that had no guardrail to catch it. This template is that guardrail.

---

## PROMPT

You are rewriting a researcher's manually-written stylized facts (about how a specific LLM
neuron's top-activating Reddit comments talk about chatbot sycophancy) into a structured
candidate-facts file. Input: `<neuron>_manual_stylized_facts.txt` — a mix of prose observations,
`id: <doc_id>: "<quote>"` citations, and pasted raw sample dumps. Output: a JSON list of
`{"fact_id", "stylized_fact", "evidence_criterion", "seed_examples"}` objects.

These candidate facts get mechanically checked against hundreds of comments by a separate
labelling pass (`label_facts.py`) that has no access to your judgment — so precision here is not
optional polish, it's the only thing standing between a real observation and a fact that gets
mislabelled at scale.

### Rule 1 — preserve polarity and claim shape. Do not invert or substitute the claim.

This is the rule that exists because it was already broken once. A manual bullet said:

> "Users want their models to feel 'natural'"

— a claim about what users *want* (a desire/preference claim) — and it got rewritten as:

> f4: "Commenters describe model output as unnatural, forced, stilted, or formulaic."

— a claim about what users *observe/complain about* (a description/complaint claim). Same
surface topic ("natural"), opposite claim shape: wanting X is not the same claim as reporting
the absence of X. That rewrite is a bug, not a style choice.

Before finalizing each candidate fact, re-read the original manual bullet side by side and check
all three of: **subject** (who/what the claim is about), **claim type** (desire vs. observation
vs. complaint vs. argument), and **direction** (for vs. against). If any of the three don't match,
rewrite again — do not accept "close enough, same general topic."

### Rule 2 — `evidence_criterion` is one operational sentence, with explicit exclusions.

It must be something a second reader (or a model with no other context) could mechanically apply
to an arbitrary comment. State what the comment must actually say, not what it's about. Follow
the existing style already used well in this repo's own candidate facts, e.g.:

> "The comment explicitly expresses intent, temptation, or a threat to cancel/unsubscribe or
> close their account/membership because of dissatisfaction with the model. Does not count if
> cancellation or closing the account is not mentioned, even if the comment expresses strong
> frustration."

Keep the "Does not count if..." exclusion clause pattern (already present in several existing
facts) — don't invent a different phrasing convention.

### Rule 3 — apply the existence restriction and the regex test (same bar as the rest of this repo).

These stylized facts follow Hirschman's sense: simple empirical regularities, offered as things
that need explaining, stated without quantifying prevalence. Concretely:
- **No quantifiers.** Banned: most / many / few / majority / minority / substantial /
  significant / common / rare / widespread / typical / usually / often / frequently /
  increasingly / rose / fell / more likely / percentages / "because" / "due to" / "led to".
  Write "Commenters do X", not "Some commenters do X" or "Many commenters do X" — existence, not
  quantity.
- **The regex test.** If `evidence_criterion` could be implemented as a keyword match (e.g.
  "the comment contains the word 'lobotomy'"), you've written a vocabulary observation, not a
  stylized fact — state what commenters CLAIM or ARGUE, not which words they use to say it.
  Vocabulary can be evidence for a fact; it isn't the fact itself.

### Rule 4 — attach `seed_examples` with full text looked up from the corpus, not the `.txt` file's partial quotes.

For each candidate fact, carry over 1–2 of the manual bullet's cited `doc_id`s as:
```json
"seed_examples": [{"doc_id": "os4yima", "text": "<full body text>"}]
```
Look up the **full, untruncated** body text by `id` from
`data/r_chatGPT_comments_relating_sycophan_2022-12-22_2026-8-7.jsonl` — do not reuse whatever
partial quote is pasted in the `.txt` file, since that file only ever captured a fragment. These
seed examples get used downstream as few-shot grounding in the labelling prompts and as a recall
sanity-check against docs you (the researcher) already confirmed are real positives, so they need
to actually be the comment that supports the fact, in full.

### Rule 5 — no invented nuance.

Only rewrite what the manual `.txt` file states. Do not add exclusions, specificity, or
refinements that aren't already present in the manual bullet, even if a raw sample dump elsewhere
in the `.txt` file suggests a plausible tightening — that's a separate editorial decision for the
researcher to make deliberately, not something to slip in during a rephrasing pass.

### Rule 6 — self-check line per fact, before returning the output.

For every candidate fact, first output one line:
```
Manual source: "<original manual bullet, verbatim or near-verbatim>" -> Candidate: "<stylized_fact>"
```
so the polarity/claim-shape check in Rule 1 is visible and reviewable, not just asserted. Then
output the final JSON.

### Output schema

```json
[
  {
    "fact_id": "f1",
    "stylized_fact": "...",
    "evidence_criterion": "...",
    "seed_examples": [{"doc_id": "...", "text": "..."}]
  }
]
```
`extract_field`/`extract_field_description` (optional, for facts where a "yes" should also
capture a short trait/value string) follow the same convention as existing candidate facts — see
`data/39774/candidate_facts.json` for a worked style reference on criterion phrasing.
