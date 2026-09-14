# Evaluating RAG systems

## Why retrieval is measured separately

If the answer is wrong, the cause is either retrieval or synthesis. Measuring
only end-to-end answer quality leaves you unable to tell which half broke, so
retrieval gets its own metrics: recall@k tells you whether the right chunk was
in the context at all, and mean reciprocal rank tells you how close to the top
it landed. If recall@5 is low, no amount of prompt engineering saves the answer.

## Golden datasets

A golden dataset is only useful if the cases come from real traffic — the
questions users actually asked, especially the ones that failed. Synthetic
question sets generated from the documents themselves grade the system on
exactly the phrasing it already handles, so the harness looks green while
production keeps failing.

Every regression found in production should become a permanent golden case.
That way the suite grows in the direction of the system's real weaknesses.

## LLM-as-judge

For open-ended answers, exact match is meaningless. An LLM judge scoring
faithfulness (is every claim supported by the retrieved context?) and relevance
(does it answer the question asked?) correlates well with human review, provided
the judge sees the context and is asked for a rubric score rather than a vibe.

Judges drift, so pin the judge model version and keep a small human-labelled set
to audit the judge itself.

## Gates, not dashboards

Metrics that do not block a release get ignored. The useful pattern is a
promotion gate: a change ships only if recall@5 and answer faithfulness stay
within an error budget of the previous build, and p95 latency does not regress.
