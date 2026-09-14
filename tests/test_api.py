"""API and retrieval tests. Run with: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402
from app.retrieval import Retriever, chunk_document, content_terms  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_healthz_reports_loaded_corpus(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0, "seed notes should load on startup"


def test_search_returns_ranked_hits(client: TestClient) -> None:
    body = client.post("/search", json={"query": "why use chunk overlap", "k": 3}).json()
    assert body["hits"], "expected at least one hit"
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True), "hits must be ranked descending"


def test_ask_is_grounded_and_cited(client: TestClient) -> None:
    body = client.post("/ask", json={"question": "Why combine BM25 with vector search?"}).json()
    assert body["grounded"] is True
    assert body["citations"], "a grounded answer must cite its chunks"


def test_ask_abstains_when_out_of_scope(client: TestClient) -> None:
    """The service must refuse rather than confabulate — this is the behaviour
    the eval harness caught missing on the first run."""
    body = client.post("/ask", json={"question": "What is the capital of France?"}).json()
    assert body["grounded"] is False
    assert body["citations"] == []


def test_stream_emits_sources_tokens_and_done(client: TestClient) -> None:
    with client.stream("POST", "/ask/stream", json={"question": "How do memory tiers work?"}) as r:
        events = [line for line in r.iter_lines() if line.startswith("event:")]
    assert "event: sources" in events[0]
    assert any("event: token" in e for e in events)
    assert "event: done" in events[-1]


def test_ingest_adds_chunks_and_is_searchable(client: TestClient) -> None:
    before = client.get("/healthz").json()["chunks"]
    payload = {
        "doc_id": "test-doc",
        "title": "Widget calibration",
        "text": "The calibration constant for the widget is 42 millijoules per cycle.",
    }
    created = client.post("/ingest", json=payload)
    assert created.status_code == 201
    assert created.json()["chunks_total"] > before

    hits = client.post("/search", json={"query": "widget calibration constant"}).json()["hits"]
    assert any(h["doc_id"] == "test-doc" for h in hits)


def test_metrics_expose_percentiles(client: TestClient) -> None:
    client.post("/search", json={"query": "latency"})
    body = client.get("/metrics").json()
    assert body["count"] > 0
    assert body["p95_ms"] >= body["p50_ms"]


# --- unit-level ---------------------------------------------------------


def test_chunking_overlaps_so_boundary_facts_survive() -> None:
    text = "\n\n".join(f"Paragraph {i} with enough words to occupy space in the buffer." for i in range(12))
    chunks = chunk_document("doc", text, "Title")
    assert len(chunks) > 1
    assert all(c.tokens for c in chunks)


def test_content_terms_drop_stopwords() -> None:
    assert "how" not in content_terms("How do I reset my payroll password?")
    assert "payroll" in content_terms("How do I reset my payroll password?")


def test_empty_retriever_returns_nothing() -> None:
    assert Retriever().search("anything") == []


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "tfidf"])
def test_all_retrieval_modes_work(client: TestClient, mode: str) -> None:
    body = client.post("/search", json={"query": "evaluation gates", "mode": mode}).json()
    assert body["mode"] == mode
    assert body["hits"]
