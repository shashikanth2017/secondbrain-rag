# Serving, latency and cost

## Perceived latency

Users complain about the wait before the first visible token, not total
generation time. Streaming partial output while tool calls are still resolving
changes the experience far more than shaving a few hundred milliseconds off the
total, because the interface stops looking frozen.

Track p50, p95 and p99 separately. An average hides the tail, and the tail is
what users remember and what pages the on-call engineer.

## Two-model routing

Most requests in an agent pipeline are not reasoning problems. Classification,
extraction, routing and summarization are high volume and comparatively easy, so
sending them to a small model and reserving the frontier model for genuine
reasoning steps cuts cost substantially without measurable quality loss — as
long as the routing decision itself is evaluated on golden cases before rollout.

The failure mode to watch: routing too aggressively to the small model degrades
quality on ambiguous inputs, and that degradation is invisible without an
evaluation gate.

## Context compression

Replaying whole conversation histories verbatim wastes input tokens and slows
time-to-first-token. Summarizing older turns and tool results keeps the prompt
small. Compression is lossy, so keep raw records of anything that must be
auditable, and compress only the working context.

## Queue-backed tool execution

Tool calls that hit slow third-party systems should run through a durable queue
with retries and timeouts. A failing downstream then degrades one step instead
of stalling the whole request, and no request is lost when a worker restarts.
