# Retrieval design notes

## Chunking

Chunk size is the first quality lever, not the embedding model. Paragraph-aware
chunking with a target of roughly 120 tokens keeps a single idea inside a single
chunk. Fixed-width splitting at 512 tokens buries the answer sentence among
unrelated text, which lowers the similarity score for the chunk that actually
contains the answer.

Overlap of about 30 tokens exists for one reason: facts that straddle a chunk
boundary. Without overlap, a sentence split across two chunks is retrievable
from neither, because each half carries only part of the signal.

## Hybrid retrieval

Lexical search and semantic search fail in different directions. BM25 misses
paraphrases — a query asking about "cost per request" will not match a note that
says "token spend" — while pure vector search misses exact identifiers such as
error codes, product names, and version numbers, because embeddings smooth away
the rare tokens that make those strings distinctive.

Running both and fusing the rankings covers each method's blind spot. Reciprocal
rank fusion is the practical choice because it operates on ranks rather than raw
scores, so the two systems need no score calibration and no tuned weight.

## Reranking

A cross-encoder reranker over the top 50 candidates usually buys more accuracy
than swapping the embedding model, because it scores the query and passage
together instead of comparing two independently computed vectors. The cost is
latency, so rerank a shortlist, never the full corpus.
