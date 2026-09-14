#!/usr/bin/env python3
"""Evaluation harness: retrieval quality, answer grounding, abstention, latency.

Run:
    python -m evals.run_eval                 # human-readable report
    python -m evals.run_eval --json          # machine-readable, for CI
    python -m evals.run_eval --gate          # exit 1 if below thresholds
    python -m evals.run_eval --compare-modes # hybrid vs bm25 vs tfidf

Every number the README quotes comes from this script. If you change retrieval,
rerun it and paste the new numbers — an unmeasured claim is not a claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.answer import ExtractiveSynthesizer  # noqa: E402
from app.retrieval import Retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "data" / "notes"
GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"

# Promotion gates. A build that drops below any of these does not ship.
THRESHOLDS = {"recall_at_5": 0.85, "mrr": 0.70, "abstention_rate": 1.0, "p95_ms": 50.0}


def load_corpus() -> Retriever:
    retriever = Retriever()
    for path in sorted(NOTES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        retriever.add_document(path.stem, text, title)
    return retriever


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((p / 100) * (len(ordered) - 1)))
    return round(ordered[idx], 2)


def evaluate(mode: str = "hybrid", k: int = 5) -> dict:
    retriever = load_corpus()
    synthesizer = ExtractiveSynthesizer()
    cases = load_golden()

    in_scope = [c for c in cases if not c.get("expect_abstain")]
    out_scope = [c for c in cases if c.get("expect_abstain")]

    hits_at_1 = hits_at_3 = hits_at_5 = 0
    reciprocal_ranks: list[float] = []
    grounded_ok = 0
    latencies: list[float] = []
    failures: list[dict] = []

    for case in in_scope:
        start = time.perf_counter()
        results = retriever.search(case["question"], k=k, mode=mode)
        answer = synthesizer.synthesize(case["question"], results)
        latencies.append((time.perf_counter() - start) * 1000)

        retrieved_docs = [h.doc_id for h in results]
        relevant = set(case["relevant_docs"])

        rank = next((i + 1 for i, d in enumerate(retrieved_docs) if d in relevant), None)
        if rank:
            reciprocal_ranks.append(1.0 / rank)
            hits_at_1 += rank <= 1
            hits_at_3 += rank <= 3
            hits_at_5 += rank <= 5
        else:
            reciprocal_ranks.append(0.0)
            failures.append(
                {
                    "id": case["id"],
                    "reason": "no relevant doc retrieved",
                    "question": case["question"],
                }
            )

        needles = [n.lower() for n in case.get("must_include", [])]
        context = " ".join(h.text.lower() for h in results)
        if all(n in context for n in needles):
            grounded_ok += 1
        elif rank:
            failures.append(
                {
                    "id": case["id"],
                    "reason": "retrieved doc lacks required evidence",
                    "question": case["question"],
                }
            )

    abstained = 0
    for case in out_scope:
        results = retriever.search(case["question"], k=k, mode=mode)
        answer = synthesizer.synthesize(case["question"], results)
        if not answer.grounded:
            abstained += 1
        else:
            failures.append(
                {
                    "id": case["id"],
                    "reason": "answered an out-of-scope question",
                    "question": case["question"],
                }
            )

    n = len(in_scope)
    return {
        "mode": mode,
        "corpus_chunks": retriever.size,
        "cases_total": len(cases),
        "cases_in_scope": n,
        "cases_out_of_scope": len(out_scope),
        "recall_at_1": round(hits_at_1 / n, 3),
        "recall_at_3": round(hits_at_3 / n, 3),
        "recall_at_5": round(hits_at_5 / n, 3),
        "mrr": round(statistics.mean(reciprocal_ranks), 3),
        "evidence_in_context": round(grounded_ok / n, 3),
        "abstention_rate": round(abstained / len(out_scope), 3) if out_scope else 1.0,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "mean_ms": round(statistics.mean(latencies), 2),
        "failures": failures,
    }


def check_gates(results: dict) -> list[str]:
    breaches = []
    for metric, floor in THRESHOLDS.items():
        value = results.get(metric)
        if value is None:
            continue
        if metric.endswith("_ms"):
            if value > floor:
                breaches.append(f"{metric}={value} exceeds ceiling {floor}")
        elif value < floor:
            breaches.append(f"{metric}={value} below floor {floor}")
    return breaches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--gate", action="store_true", help="exit non-zero if thresholds breached")
    parser.add_argument("--compare-modes", action="store_true")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "tfidf"])
    args = parser.parse_args()

    if args.compare_modes:
        table = [evaluate(m) for m in ("bm25", "tfidf", "hybrid")]
        if args.json:
            print(json.dumps(table, indent=2))
            return 0
        print(f"{'mode':<8}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'MRR':>7}{'p95 ms':>9}")
        for row in table:
            print(
                f"{row['mode']:<8}{row['recall_at_1']:>7}{row['recall_at_3']:>7}"
                f"{row['recall_at_5']:>7}{row['mrr']:>7}{row['p95_ms']:>9}"
            )
        return 0

    results = evaluate(args.mode)
    breaches = check_gates(results)

    if args.json:
        print(json.dumps({**results, "gate_breaches": breaches}, indent=2))
    else:
        print(f"\nSecondBrain RAG — evaluation ({results['mode']} retrieval)")
        print(f"  corpus: {results['corpus_chunks']} chunks | cases: {results['cases_total']} "
              f"({results['cases_in_scope']} in-scope, {results['cases_out_of_scope']} out-of-scope)\n")
        print(f"  recall@1            {results['recall_at_1']:.3f}")
        print(f"  recall@3            {results['recall_at_3']:.3f}")
        print(f"  recall@5            {results['recall_at_5']:.3f}")
        print(f"  MRR                 {results['mrr']:.3f}")
        print(f"  evidence in context {results['evidence_in_context']:.3f}")
        print(f"  abstention (o.o.s.) {results['abstention_rate']:.3f}")
        print(f"  latency p50/p95/p99 {results['p50_ms']} / {results['p95_ms']} / {results['p99_ms']} ms")
        if results["failures"]:
            print("\n  failures:")
            for f in results["failures"]:
                print(f"    - [{f['id']}] {f['reason']}: {f['question']}")
        print("\n  gates:", "PASS" if not breaches else "FAIL")
        for b in breaches:
            print(f"    ! {b}")

    return 1 if (args.gate and breaches) else 0


if __name__ == "__main__":
    raise SystemExit(main())
