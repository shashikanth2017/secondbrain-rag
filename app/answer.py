"""Answer synthesis with grounding guarantees.

Two backends behind one interface:

* `ExtractiveSynthesizer` (default) — no API key, fully deterministic. Selects
  supporting sentences from retrieved chunks. Every sentence it emits exists
  verbatim in the corpus, so the answer is grounded by construction and the
  eval harness measures retrieval quality without an LLM in the loop.
* `LLMSynthesizer` — used when OPENAI_API_KEY is set. Same context contract,
  same citation format, so swapping backends does not change the API surface.

The abstention rule matters more than the model: if retrieval returns nothing
above the confidence floor, the service says it does not know. In a
knowledge-base assistant a confident wrong answer costs far more than a refusal.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .retrieval import Hit

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "on", "with", "how", "what", "why", "when", "which", "do", "does",
    "did", "i", "my", "it", "that", "this", "as", "at", "by", "be", "from",
}


@dataclass
class Answer:
    text: str
    citations: list[str]
    grounded: bool
    backend: str


def _keywords(question: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOPWORDS}


class ExtractiveSynthesizer:
    """Deterministic, offline, and grounded by construction."""

    backend = "extractive"

    def __init__(self, min_score: float = 0.0, max_sentences: int = 3) -> None:
        self.min_score = min_score
        self.max_sentences = max_sentences

    def synthesize(self, question: str, hits: list[Hit]) -> Answer:
        usable = [h for h in hits if h.score > self.min_score]
        if not usable:
            return Answer(
                text="I don't have anything in the knowledge base that answers that.",
                citations=[],
                grounded=False,
                backend=self.backend,
            )

        keywords = _keywords(question)
        scored: list[tuple[float, str, str]] = []
        for hit in usable[:3]:
            for sentence in SENTENCE_RE.split(hit.text):
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(keywords & words)
                if overlap:
                    scored.append((overlap / (len(keywords) or 1), sentence, hit.chunk_id))

        if not scored:
            best = usable[0]
            first = SENTENCE_RE.split(best.text)[0].strip()
            return Answer(text=first, citations=[best.chunk_id], grounded=True, backend=self.backend)

        scored.sort(key=lambda t: t[0], reverse=True)
        picked, citations, seen = [], [], set()
        for _, sentence, chunk_id in scored[: self.max_sentences]:
            if sentence in seen:
                continue
            seen.add(sentence)
            picked.append(sentence)
            if chunk_id not in citations:
                citations.append(chunk_id)

        return Answer(text=" ".join(picked), citations=citations, grounded=True, backend=self.backend)


class LLMSynthesizer:
    """OpenAI-backed synthesis. Same contract; used only when a key is present."""

    backend = "llm"

    def __init__(self, model: str = "gpt-4o-mini", max_context_chars: int = 6000) -> None:
        self.model = model
        self.max_context_chars = max_context_chars

    def synthesize(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(
                text="I don't have anything in the knowledge base that answers that.",
                citations=[],
                grounded=False,
                backend=self.backend,
            )
        from openai import OpenAI  # imported lazily: optional dependency

        context, used = [], []
        budget = self.max_context_chars
        for hit in hits:
            block = f"[{hit.chunk_id}] {hit.text}"
            if len(block) > budget:
                break
            context.append(block)
            used.append(hit.chunk_id)
            budget -= len(block)

        client = OpenAI()
        completion = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided context. Cite the chunk ids you used "
                        "in square brackets. If the context does not contain the answer, say "
                        "you don't know. Never invent facts."
                    ),
                },
                {"role": "user", "content": f"Context:\n{chr(10).join(context)}\n\nQuestion: {question}"},
            ],
        )
        text = completion.choices[0].message.content or ""
        return Answer(text=text, citations=used, grounded=True, backend=self.backend)


def get_synthesizer():
    """LLM backend when a key exists, deterministic extractive otherwise."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return LLMSynthesizer()
        except (ImportError, ValueError):  # pragma: no cover - optional dependency
            return ExtractiveSynthesizer()
    return ExtractiveSynthesizer()
