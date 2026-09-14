# Agent architecture notes

## Tool schemas

Models follow narrow, strongly typed tool schemas far more reliably than broad
ones. A tool taking a free-form `query` string invites malformed calls; the same
tool with explicit enum fields and required parameters is called correctly far
more often. Schema design is prompt engineering by another name.

Validate every tool result against the schema before it re-enters the context.
A malformed tool response that reaches the model produces a confidently wrong
answer that is hard to trace afterwards.

## Termination

Two agents that can message each other will message each other forever unless a
protocol stops them. Practical guards: a hard step budget per task, a rule that
an agent may not reply to its own delegate more than N times, and an explicit
completion signal that ends the exchange.

## Memory tiers

Working memory holds the current task, episodic memory holds recent
interactions, and semantic memory holds durable facts retrieved on demand.
Collapsing all three into one long prompt is what makes agents both expensive
and forgetful — the prompt grows without bound while the relevant fact gets
buried.

## Non-determinism

The same input can produce different tool call sequences across runs. Systems
built on agents therefore need idempotent tools, retries that are safe to repeat,
and evaluation over multiple runs rather than a single pass.
