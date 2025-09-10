# Backlog Buckets

## NOW (WIP ≤ 2)

1) Core RAG — Hybrid Retrieval + Reranker MVP
   - Why it matters (metric impact): Retrieval quality is the #1 lever for grounded, useful answers.
   - Acceptance:
     - Eval set: 50 real/synthetic queries (lookup/procedural/escalate mix).
     - P@5 ≥ 70% on dev set; nDCG@10 reported.
     - p95 end-to-end ≤ 1.6s with semantic cache enabled.
     - README shows before/after and trace screenshot.

2) Supporting Agent/Ops — Ticket Draft Tool (Dry-Run) + Tracing/Cost
   - Why it matters (metric impact): Turns answers into action; demos “closing the loop”.
   - Acceptance:
     - `open_ticket(json)` tool fills fields from answer JSON (no external write).
     - Trace includes retrieve → rerank → tool call with inputs/outputs.
     - Cost/query and p95 latency logged to `/ops/metrics.md`.
     - 90s demo script prepared.

---

## NEXT (2–4 items; justified by eval gaps or trace pain)

1) Multimodal Intake — Photo → Part/Spec Link
   - Why it matters (metric impact): Reduces part mis-ID; unlocks field-service value.
   - Acceptance:
     - 30 labeled photos; top-3 ≥ 80% on dev set.
     - Answer cites matching spec doc anchor.
     - Trace shows vision retrieval step.

2) Guardrails — Injection/PII + CI Gates
   - Why it matters (metric impact): Prevents regressions and unsafe outputs; interview-ready reliability.
   - Acceptance:
     - Block pasted-URL instructions & code-block jailbreak patterns.
     - PII redaction in logs/traces.
     - CI fails if P@5 drops >3 pts or p95 exceeds budget.

3) Freshness/Versioning — Doc Invalidation
   - Why it matters (metric impact): Keeps answers aligned with latest SOPs; reduces stale-citation rate.
   - Acceptance:
     - Doc lineage (hash/version) stored per chunk.
     - New version triggers index invalidation within 5 minutes.
     - Metric: stale-citation rate tracked weekly.

4) Model Router — SLM→LLM Fallback
   - Why it matters (metric impact): Latency/cost improvements without quality loss.
   - Acceptance:
     - Classifier routes “easy” lookups to SLM, “hard” to LLM.
     - Cost/query reduced ≥ 30% with P@5 within −1 pt of baseline.

---

## LATER (parking lot; tie to KPI hypothesis)

1) Inventory Tool (Read-Only) + Human Approval
   - KPI hypothesis: Lowers time-to-action on parts by 15% (proxy in demo).

2) JSON Checklist Outputs for SOP Steps
   - KPI hypothesis: Cuts novice error rate; measure “checklist completeness” in eval assertions.

3) Slack/Email Summarizer with Citations + Trace Link
   - KPI hypothesis: Improves cross-team handoff speed; track response time to first action.

4) On-device Semantic Cache / Edge Embeddings (exploratory)
   - KPI hypothesis: p95 −20% for common queries; measure cache hit rate.

