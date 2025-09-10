# Daily & Weekly Workflows

## Daily (`@dailystart`)
**Today**
- Build: <feature> (≤90m)
- Support: <agent/ops step> (≤60m)
- Metric to move: <e.g., P@5 +5 pts>

**Plan (≤35m)**
- Define 2–4 interfaces only
- Write 3 evals that fail if we’re wrong
- Acceptance tests for both tasks

**Build (≤90m)**
- Happy path end-to-end; mock deps if needed

**Instrument (20m)**
- Add a trace, one test, and metric logging

**Decide (10m)**
- Keep/simplify/kill; record 20–40s demo clip

## Weekly Balance (RAG + Agent + Eval/Ops)
- Week 0: scaffold repo, tracing, Docker, seed (10 docs, 5 queries)
- Week 1: Hybrid + reranker, ticket draft tool, 50 evals + CI
- Week 2: Multimodal intake, escalation helper, injection/PII tests + cost logging
- Week 3: Freshness/versioning, inventory read-only + approvals, p95 < 1.5s
- Week 4: JSON checklists, Slack/Email summarizer, trend board + drift alert
(Repeat with thin slices; maintain triad weekly.)

## Friday (`@friday`)
- 90s Demo script ready
- Metrics: before vs after (P@5, p95, cost)
- One trace screenshot + trade-off
- 3 bullets: learnings
- Next week’s single bottleneck
