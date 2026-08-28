import re

# banning because these are non-existence stylized facts
BANNED_PATTERNS = [
    r"\bmost\b", r"\bmany\b", r"\bfew\b", r"\bmajority\b", r"\bminority\b",
    r"\bsubstantial\b", r"\bsignificant(?:ly)?\b", r"\bcommon(?:ly)?\b", r"\brare(?:ly)?\b",
    r"\bwidespread\b", r"\btypical(?:ly)?\b", r"\busually\b", r"\boften\b",
    r"\bfrequent(?:ly)?\b", r"\bincreas(?:e|ed|ing)\b", r"\bdecreas(?:e|ed|ing)\b",
    r"\brose\b", r"\bfell\b", r"\bmore likely\b", r"\bless likely\b",
    r"\bcaus(?:e|ed|es|ing)\b", r"\bbecause of\b", r"\bdue to\b", r"\bled to\b",
    r"\d+\s*%", r"\bpercent\b", r"\bproportion\b", r"\bshare of\b",
]
BANNED_RE = re.compile("|".join(BANNED_PATTERNS), re.I)

TARGET_BLURBS = {
    "gpt5_marker": (
        "Features were selected by contrasting comments posted on/after the GPT-5 release "
        "(2025-08-07 UTC) with comments posted before it. 'positive' = fires more after; "
        "'negative' = fires more before. A contrast with calendar time, not a causal claim "
        "about the release."),
    "engage": (
        "Features were selected by contrasting comments with high vs low within-thread "
        "standardised log score (Reddit upvotes). 'positive' = fires on comments scoring "
        "above their thread's mean."),
    "ontopic": (
        "Features were selected by contrasting comments that literally contain a sycophancy "
        "lexicon term (sycophan*, glaz*, flatter*, yes-man, kiss-ass, brown-nos*, boot-lick*, "
        "pander, suck-up) with comments in the same corpus that do not. 'positive' = fires on "
        "lexicon-bearing comments. Note this is close to a keyword detector."),
}

CORPUS_BLURB = (
    "All texts are Reddit comments from r/ChatGPT, 2022-12-22 to 2026-08-07. The corpus was "
    "pre-filtered: a comment is included if it contains 'sycophan' OR is attached to a "
    "submission containing 'sycophan'. The entire corpus is therefore already "
    "sycophancy-adjacent. Any fact that amounts to 'people here discuss AI flattery' is "
    "vacuous and will be rejected."
)

GEN_SYSTEM = """You are an empirical social scientist doing exploratory analysis of a text corpus.
 
A sparse autoencoder ran over embeddings of the corpus. Someone else already selected the
SAE features that most separate a target variable. You read the top-activating comments of
those features and state, in plain language, what recurring thing is present in them.

THE REGEX TEST -- apply this to every fact before you write it down.
If your evidence_criterion could be implemented as a keyword or regular-expression
match over the comment text, the fact is an observation about VOCABULARY, not a
stylized fact. "Commenters use the word X", "commenters link to Y", "commenters
quote Z" all fail this test: a five-token regex reproduces them, and nothing about
them needs explaining.
State what commenters CLAIM, ARGUE, ASSUME or REPORT -- a position someone could
disagree with -- not which words they use to say it. The criterion should require a
reader to understand what the comment is asserting, not just to spot a string.
Vocabulary may be evidence FOR a fact; it may not be the fact.
 
WHAT YOU ARE WRITING
Stylized facts in Hirschman's (2016) sense: simple empirical regularities stated in
non-specialist categories, offered as things that need explaining, agnostic about why.
In this pass you are under two additional constraints.
 
CONSTRAINT 1 -- THE EXISTENCE RESTRICTION
Every fact must be fully verifiable by reading the supporting comments alone. A reader
holding only those comments must be able to say "yes, that is there" without counting,
without comparing rates, and without trusting the sample to be representative.
So each fact asserts that an identifiable position, complaint, framing, rhetorical move or
vocabulary is PRESENT among commenters, and characterises it specifically. It must NOT
assert how prevalent it is, whether it grew or shrank, whether it is more or less common
than something else, or what caused it.
BANNED: most / many / few / majority / minority / substantial / significant / common /
rare / widespread / typical / usually / often / frequently / increasingly / rose / fell /
more likely / percentages / "because" / "due to" / "led to". Do not smuggle prevalence in
with hedges ("a notable number", "a recurring share"). "Some commenters" is banned too --
write "Commenters do X", which asserts existence without quantity.
The association is NOT your job: it is already carried by each feature's z and direction,
which are recorded separately. Do not put it in the sentence.
 
CONSTRAINT 2 -- THE SPANNING OBJECTIVE
You are shown K features. Attribute each fact to AT MOST K of them. Two things are traded
off and you must hold both:
  (a) SPAN. Prefer facts that genuinely hold across several of the K features rather than
      facts describing one feature's quirk. A fact attributed to 4 features beats an
      equally specific fact attributed to 1.
  (b) SUPPORT. Prefer facts that a large share of those features' top comments actually
      satisfy. A fact only 3 comments instantiate is a specific fact, not a regularity.
  BUT (c) is a hard veto on both: the fact must stay specific enough that a comment drawn
      from elsewhere in this corpus would FAIL its criterion. The corpus is pre-filtered on
      sycophancy, so broadening a fact until it spans everything produces something that
      matches the whole corpus and tells the reader nothing. That is the failure mode being
      guarded against. When span and specificity conflict, keep specificity and attribute to
      fewer features.
Do not attribute a fact to a feature merely because that feature is in the list. Attribute
it only where you can see the fact instantiated in that feature's comments.
 
OTHER REQUIREMENTS
  - Facts must be mutually distinct. Do not restate one fact at two levels of specificity.
  - Ground facts in the comments shown, not in what you know about ChatGPT from elsewhere.
  - Do not describe the SAE, features, activations or this prompt in the fact text.
  - For each fact write `evidence_criterion`: one sentence, mechanically applicable, that a
    second reader could apply to an arbitrary comment to decide support. Operational (what
    must the comment actually say?), not evaluative. It is used verbatim in a later pass, so
    it must be self-contained -- it may not refer to "the fact above" or to feature numbers.
 
WORKED EXAMPLES
GOOD  "Commenters invert the standard complaint, arguing that GPT-5's refusal to flatter
       makes it feel more emotionally engaging rather than colder."
GOOD  "Commenters police the word 'sycophancy' itself, accusing other posters of misapplying
       it to ordinary politeness or empathy."
BAD   "A substantial post-launch minority reports GPT-5.x is more sycophantic than 4o."
       -> quantified; unverifiable from examples.
BAD   "Commenters discuss ChatGPT's tendency toward flattery."
       -> vacuous: matches the entire pre-filtered corpus. Maximal span, zero content.
BAD   "GPT-5's launch caused users to switch to Claude."
       -> causal.
 
OUTPUT
Return ONLY a JSON object, no prose, no markdown fence:
{"facts": [{"fact_id": "f1",
            "stylized_fact": "...",
            "evidence_criterion": "...",
            "feature_ids": [12345, 67890]}, ...]}
`feature_ids` must be drawn from the feature ids shown to you."""

ATTR_SYSTEM = """You label Reddit comments against a fixed list of claims.
 
For each comment, decide which claims it SUPPORTS. A comment supports a claim only if the
comment's own text satisfies that claim's evidence criterion. Be strict:
  - Topical adjacency is not support. The comment must instantiate the claim.
  - Do not infer the author's unstated views.
  - A comment may support zero, one, or several claims.
  - Sarcasm counts only if the criterion concerns a stance and the sarcastic reading clearly
    expresses that stance.
 
Return ONLY a JSON object, no prose, no markdown fence:
{"labels": [{"comment_id": 123, "supports": ["f1","f4"]}, ...]}
Include every comment id you were given, with an empty list if it supports nothing."""

REPAIR_SYSTEM = GEN_SYSTEM + """
 
RECALIBRATION PASS
You previously wrote these facts. Each now carries what happened when its criterion was
applied to real comments:
 
  support        how many of its features' top comments satisfied the criterion, out of
                 how many were eligible
  per-feature    the same count broken out by feature
  precision      of everything the criterion matched, the share drawn from comments where
                 the features actually fire, rather than from comments where they do NOT.
                 LOW PRECISION MEANS THE FACT IS TOO BROAD: it is matching the general
                 corpus rather than the pattern.
  false positives  comments where the features do not fire that the criterion matched anyway
  near misses      comments from the fact's own features that the criterion did NOT match
 
YOUR JOB IS CALIBRATION, NOT MAXIMISATION. Low support is not automatically a defect. A
fact can be correct and rare. Diagnose which of these is true and act accordingly:
 
  (a) OVER-FITTED -- the fact describes something that appears once or twice because it was
      generalised from a single comment. The near misses show the feature is about a
      broader recurring pattern that this fact names too narrowly. Restate the fact at the
      level that actually recurs.
  (b) DRIFTED -- the criterion is looser than the fact, so it matched comments the fact does
      not actually claim. Tighten the criterion to match the fact's literal assertion.
  (c) TOO BROAD -- precision is low; the criterion matches comments where the feature does
      not fire. Narrow it. LOSING EXAMPLES HERE IS THE CORRECT OUTCOME.
  (d) CORRECT AS IS -- the near misses genuinely do not instantiate the fact, and precision
      is high. Return the fact unchanged. This is a legitimate and expected answer, including
      when support is in the single digits.
 
FORBIDDEN: you may not raise support by restating what the feature is about. "Commenters
compare model versions", "commenters discuss flattery" and similar topical restatements are
vacuous -- they would match nearly every comment in a corpus that was already filtered on
this topic. If the only way to increase support is to broaden toward a topic label, choose
(d) and leave the fact alone.
 
Keep every fact_id exactly as it was, and return the same number of facts you were given, so
results stay comparable across runs. You may change stylized_fact, evidence_criterion and
feature_ids. Same output schema as before."""
