"""FastAPI service: ingest notes, search them, ask grounded questions.

Endpoints intentionally mirror the shape production RAG services converge on:
/ingest, /search, /ask, /ask/stream, /healthz, /metrics.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .answer import get_synthesizer
from .retrieval import Retriever

retriever = Retriever()
synthesizer = get_synthesizer()
_latencies: deque[float] = deque(maxlen=1000)

NOTES_DIR = Path(__file__).resolve().parent.parent / "data" / "notes"


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_seed_notes()
    yield


app = FastAPI(
    title="SecondBrain RAG API",
    version="1.0.0",
    description="Retrieval-augmented question answering over a personal knowledge base.",
    lifespan=lifespan,
)


class IngestRequest(BaseModel):
    doc_id: str = Field(..., examples=["retrieval-notes"])
    text: str = Field(..., min_length=1)
    title: str = ""


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)
    mode: str = Field("hybrid", pattern="^(hybrid|bm25|tfidf)$")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)


def load_seed_notes() -> int:
    """Load the sample corpus so a fresh clone is queryable immediately."""
    if not NOTES_DIR.exists():
        return 0
    count = 0
    for path in sorted(NOTES_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        retriever.add_document(doc_id=path.stem, text=text, title=title)
        count += 1
    return count


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "chunks": retriever.size, "backend": synthesizer.backend}


@app.get("/metrics")
def metrics() -> dict:
    """Request-latency percentiles measured in-process."""
    if not _latencies:
        return {"count": 0}
    ordered = sorted(_latencies)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, round((p / 100) * (len(ordered) - 1)))
        return round(ordered[idx], 2)

    return {
        "count": len(ordered),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": round(ordered[-1], 2),
    }


@app.post("/ingest", status_code=201)
def ingest(req: IngestRequest) -> dict:
    added = retriever.add_document(req.doc_id, req.text, req.title)
    return {"doc_id": req.doc_id, "chunks_added": added, "chunks_total": retriever.size}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    start = time.perf_counter()
    hits = retriever.search(req.query, k=req.k, mode=req.mode)
    elapsed = (time.perf_counter() - start) * 1000
    _latencies.append(elapsed)
    return {
        "query": req.query,
        "mode": req.mode,
        "took_ms": round(elapsed, 2),
        "hits": [h.__dict__ for h in hits],
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if retriever.size == 0:
        raise HTTPException(status_code=409, detail="Knowledge base is empty; POST /ingest first.")
    start = time.perf_counter()
    hits = retriever.search(req.question, k=req.k)
    answer = synthesizer.synthesize(req.question, hits)
    elapsed = (time.perf_counter() - start) * 1000
    _latencies.append(elapsed)
    return {
        "question": req.question,
        "answer": answer.text,
        "citations": answer.citations,
        "grounded": answer.grounded,
        "backend": answer.backend,
        "took_ms": round(elapsed, 2),
        "sources": [{"chunk_id": h.chunk_id, "title": h.title, "score": h.score} for h in hits],
    }


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """SSE streaming: first token leaves before the full answer is assembled."""
    if retriever.size == 0:
        raise HTTPException(status_code=409, detail="Knowledge base is empty; POST /ingest first.")

    async def event_source():
        start = time.perf_counter()
        hits = retriever.search(req.question, k=req.k)
        yield f"event: sources\ndata: {json.dumps([h.chunk_id for h in hits])}\n\n"
        answer = synthesizer.synthesize(req.question, hits)
        for token in answer.text.split(" "):
            yield f"event: token\ndata: {json.dumps(token + ' ')}\n\n"
            await asyncio.sleep(0)
        elapsed = (time.perf_counter() - start) * 1000
        _latencies.append(elapsed)
        payload = {"citations": answer.citations, "grounded": answer.grounded, "took_ms": round(elapsed, 2)}
        yield f"event: done\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
