"""Hybrid retrieval: BM25 lexical + TF-IDF dense-ish semantic, fused with RRF.

Deliberately dependency-light: no API keys, no network, no GPU. The point of this
service is to demonstrate retrieval *engineering* — chunking, hybrid fusion,
reranking and measurable quality — not to wrap a hosted embedding API. An
OpenAI embedding backend can be dropped in behind `Retriever.embed` without
touching the rest of the pipeline.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words carry no topical signal, so they must not count towards the
# in-scope check below — otherwise "How do I reset my payroll password?" looks
# answerable purely because "how", "do", "i" and "my" appear in the corpus.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but",
    "by", "can", "did", "do", "does", "for", "from", "had",
    "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "no", "not", "of", "on",
    "or", "should", "so", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "upon", "use",
    "used", "using", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
})


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    title: str = ""
    tokens: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(f"{self.title} {self.text}")


def chunk_document(
    doc_id: str,
    text: str,
    title: str = "",
    target_tokens: int = 120,
    overlap_tokens: int = 30,
) -> list[Chunk]:
    """Paragraph-aware chunking with overlap.

    Splitting on blank lines first keeps semantically coherent units together;
    only oversized paragraphs get windowed. Overlap exists so a fact sitting on
    a chunk boundary is still retrievable from at least one chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        body = "\n\n".join(buffer)
        chunks.append(
            Chunk(doc_id=doc_id, chunk_id=f"{doc_id}#{len(chunks)}", text=body, title=title)
        )
        if overlap_tokens and len(tokenize(body)) > overlap_tokens:
            tail = " ".join(body.split()[-overlap_tokens:])
            buffer, buffer_len = [tail], len(tokenize(tail))
        else:
            buffer, buffer_len = [], 0

    for para in paragraphs:
        para_len = len(tokenize(para))
        if para_len > target_tokens * 2:
            words = para.split()
            step = target_tokens
            for i in range(0, len(words), step):
                window = " ".join(words[i : i + step + overlap_tokens])
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}#{len(chunks)}",
                        text=window,
                        title=title,
                    )
                )
            continue
        if buffer_len + para_len > target_tokens:
            flush()
        buffer.append(para)
        buffer_len += para_len

    flush()
    return chunks


class BM25:
    """Okapi BM25. Written out rather than imported so the ranking is inspectable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.doc_len = [len(d) for d in corpus]
        self.avgdl = (sum(self.doc_len) / len(corpus)) if corpus else 0.0
        self.df: Counter[str] = Counter()
        for doc in corpus:
            self.df.update(set(doc))
        self.n = len(corpus)
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }
        self.tf: list[Counter[str]] = [Counter(doc) for doc in corpus]

    def score(self, query_tokens: Iterable[str]) -> list[float]:
        scores = [0.0] * self.n
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (self.doc_len[i] / (self.avgdl or 1))
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores


class TfidfIndex:
    """Cosine similarity over L2-normalised TF-IDF vectors (pure Python)."""

    def __init__(self, corpus: list[list[str]]) -> None:
        self.n = len(corpus)
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        self.idf = {t: math.log((self.n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
        self.vectors = [self._vectorize(doc) for doc in corpus]

    def _vectorize(self, tokens: Iterable[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def score(self, query_tokens: Iterable[str]) -> list[float]:
        qvec = self._vectorize(query_tokens)
        out = []
        for vec in self.vectors:
            if len(qvec) < len(vec):
                out.append(sum(w * vec.get(t, 0.0) for t, w in qvec.items()))
            else:
                out.append(sum(w * qvec.get(t, 0.0) for t, w in vec.items()))
        return out


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """RRF: rank-based fusion, so lexical and semantic scores need no calibration."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


class Retriever:
    """In-memory hybrid retriever. Rebuilds indexes on ingest (corpus is small)."""

    def __init__(self, min_term_coverage: float = 0.34) -> None:
        self.chunks: list[Chunk] = []
        self._bm25: BM25 | None = None
        self._tfidf: TfidfIndex | None = None
        self._vocab: set[str] = set()
        # Fraction of a query's content terms that must exist in the corpus before
        # the retriever will return anything. Without this floor, RRF happily
        # ranks the least-irrelevant chunk for a question the corpus cannot
        # answer, and the service confabulates instead of abstaining. The eval
        # harness (evals/run_eval.py) measures this as `abstention_rate`.
        self.min_term_coverage = min_term_coverage

    def add_document(self, doc_id: str, text: str, title: str = "") -> int:
        new_chunks = chunk_document(doc_id, text, title)
        self.chunks.extend(new_chunks)
        self._reindex()
        return len(new_chunks)

    def _reindex(self) -> None:
        corpus = [c.tokens for c in self.chunks]
        self._bm25 = BM25(corpus) if corpus else None
        self._tfidf = TfidfIndex(corpus) if corpus else None
        self._vocab = {t for doc in corpus for t in doc}

    def term_coverage(self, query: str) -> float:
        """Share of the query's content terms that exist in the corpus at all."""
        terms = content_terms(query)
        if not terms:
            return 0.0
        return sum(1 for t in terms if t in self._vocab) / len(terms)

    def in_scope(self, query: str) -> bool:
        return self.term_coverage(query) >= self.min_term_coverage

    @property
    def size(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if not self.chunks or self._bm25 is None or self._tfidf is None:
            return []
        if not self.in_scope(query):
            return []
        q = tokenize(query)
        lexical = self._bm25.score(q)
        semantic = self._tfidf.score(q)

        if mode == "bm25":
            order = sorted(range(len(lexical)), key=lambda i: lexical[i], reverse=True)
            scored = [(i, lexical[i]) for i in order]
        elif mode == "tfidf":
            order = sorted(range(len(semantic)), key=lambda i: semantic[i], reverse=True)
            scored = [(i, semantic[i]) for i in order]
        else:
            lex_rank = sorted(range(len(lexical)), key=lambda i: lexical[i], reverse=True)
            sem_rank = sorted(range(len(semantic)), key=lambda i: semantic[i], reverse=True)
            scored = reciprocal_rank_fusion([lex_rank[:50], sem_rank[:50]])

        hits: list[Hit] = []
        for idx, score in scored[:k]:
            if score <= 0:
                continue
            c = self.chunks[idx]
            hits.append(
                Hit(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    title=c.title,
                    text=c.text,
                    score=round(float(score), 6),
                )
            )
        return hits
