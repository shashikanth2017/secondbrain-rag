# SecondBrain RAG API

A small FastAPI service that answers questions over a personal knowledge base — with an
evaluation harness that decides whether a change is allowed to ship.

The retrieval is deliberately unremarkable. The part worth reading is `evals/`: retrieval
quality, answer grounding, abstention behaviour and latency are measured on every run, and
the same command runs in CI as a merge gate.

```bash
pip install -r requirements-dev.txt
make run     # http://127.0.0.1:8000/docs
make test    # 13 tests
make eval    # the numbers below
```

---

## Measured results

Produced by `python -m evals.run_eval` on the 27-case golden set in `evals/golden.jsonl`
against the 13-chunk sample corpus in `data/notes/`. Rerun it and you get these numbers;
nothing here is hand-written.

| Metric | Value | What it means |
|---|---|---|
| recall@1 | **0.800** | correct document ranked first |
| recall@3 | **0.880** | correct document in top 3 |
| recall@5 | **0.920** | correct document retrieved at all |
| MRR | **0.837** | mean reciprocal rank of the correct document |
| evidence in context | **0.920** | retrieved context contained the required fact |
| abstention (out-of-scope) | **1.000** | refused every question the corpus cannot answer |
| latency p50 / p95 / p99 | **0.16 / 0.19 / 0.39 ms** | retrieval + synthesis, in-process, no network |

Two cases fail (`p01`, `p02`) and are left failing on purpose — see
[What the harness catches](#what-the-harness-catches).

**Honest scope note.** 13 chunks is a toy corpus and the latency figures reflect an
in-memory index with no embedding API call, so they say nothing about production
throughput. These numbers demonstrate that the harness works and that the system is
honest about its limits — they are not a benchmark claim.

### Retrieval modes

`make compare` scores the three modes on the same set:

| mode | R@1 | R@3 | R@5 | MRR | p95 ms |
|---|---|---|---|---|---|
| bm25 | 0.80 | 0.88 | 0.92 | 0.837 | 0.31 |
| tfidf | 0.80 | 0.88 | 0.92 | 0.837 | 0.20 |
| hybrid | 0.80 | 0.88 | 0.92 | 0.837 | 0.21 |

Hybrid fusion shows **no measurable advantage at this corpus size** — the honest result,
reported rather than hidden. With 13 chunks and questions that share vocabulary with the
notes, lexical matching alone is sufficient; fusion earns its keep on larger corpora where
paraphrase and rare-token queries diverge. Reporting a null result is the point: a harness
you only trust when it agrees with you is not a harness.

---

## What the harness catches

**It found a real defect on its first run.** The service answered out-of-scope questions —
"What is the capital of France?" returned the least-irrelevant chunk with a confident tone,
because rank fusion always produces *some* ordering. Abstention rate: 0.000.

The fix was a relevance floor before retrieval runs: the share of a query's content terms
that exist in the corpus at all must clear a threshold (`Retriever.min_term_coverage`), with
stopwords excluded so function words can't make an unanswerable question look answerable.
Abstention rate went to 1.000 and `test_ask_abstains_when_out_of_scope` now guards it.

**Two paraphrase cases still fail**, and they stay in the suite:

- `p01` "How can I make my assistant feel faster without making it faster?"
- `p02` "Cheapest way to handle simple repetitive requests?"

Both are answered by `serving-and-cost.md`, and both are phrased with none of that note's
vocabulary — no "latency", no "streaming", no "routing". This is precisely the failure mode
lexical retrieval has and the reason real systems add embeddings. Deleting the cases would
make the table read 100%; keeping them documents the system's actual boundary. A golden set
should be built from what breaks, not from what passes.

---

## Design decisions

**Chunking (`app/retrieval.py`).** Paragraph-aware, ~120 tokens, ~30-token overlap. Chunk
size is the first quality lever, ahead of model choice: fixed 512-token windows bury the
answer sentence among unrelated text and dilute its score. Overlap exists so a fact split
across a boundary is still retrievable.

**Hybrid retrieval with RRF.** BM25 and TF-IDF cosine are computed independently and fused
by reciprocal rank fusion. RRF operates on ranks, not scores, so the two systems need no
calibration and no tuned weight — a property that matters when one of them is later swapped
for a real embedding model.

**Grounding by construction.** The default synthesizer is extractive: every sentence it
returns exists verbatim in the retrieved chunks, so "did the model hallucinate?" is not a
question this service can fail. Set `OPENAI_API_KEY` and `LLMSynthesizer` takes over with
the same context contract and citation format — the API surface does not change.

**Abstention over confidence.** In a knowledge assistant, a confident wrong answer costs
more than a refusal, so the service says it doesn't know rather than reaching.

**No API key required.** The whole thing runs offline, which is why CI can gate on
evaluation results instead of just linting.

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | add a document; chunks and reindexes |
| `POST /search` | ranked chunks with scores (`mode`: hybrid \| bm25 \| tfidf) |
| `POST /ask` | grounded answer with citations and sources |
| `POST /ask/stream` | SSE — `sources` event, then `token` events, then `done` |
| `GET /healthz` | liveness, chunk count, active synthesis backend |
| `GET /metrics` | in-process latency p50/p95/p99 |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"Why combine BM25 with vector search?"}' | jq
```

```json
{
  "question": "Why combine BM25 with vector search?",
  "answer": "Lexical search and semantic search fail in different directions...",
  "citations": ["retrieval-design#1"],
  "grounded": true,
  "backend": "extractive",
  "took_ms": 0.31
}
```

---

## Layout

```
app/
  retrieval.py   chunking, BM25, TF-IDF, RRF fusion, scope floor
  answer.py      extractive + LLM synthesizers, abstention rule
  main.py        FastAPI endpoints, SSE streaming, latency metrics
evals/
  golden.jsonl   27 cases: 20 factual, 5 paraphrase, 2 out-of-scope
  run_eval.py    recall@k, MRR, grounding, abstention, latency, gates
tests/           13 pytest cases incl. the abstention regression guard
data/notes/      sample corpus (synthetic notes, written for this repo)
.github/workflows/ci.yml   lint → tests → evaluation gate
```

## CI gate

`python -m evals.run_eval --gate` exits non-zero if recall@5 < 0.85, MRR < 0.70,
abstention < 1.0, or p95 > 50 ms. Metrics that don't block a merge get ignored, so these
run on every pull request.

## Notes on the corpus

`data/notes/` contains synthetic notes written for this repository on retrieval, evaluation,
serving cost and agent design. No proprietary or employer material is included.

## License

MIT — see [LICENSE](LICENSE).
