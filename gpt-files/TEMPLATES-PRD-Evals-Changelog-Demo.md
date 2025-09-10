# Templates

## PRD-Lite.md
**Feature:** <name>  
**Problem:** <who hurts and why>  
**Users:** <roles>  
**Success Metric:** <one number, e.g., P@5 +7 pts or p95 <1.5s>  
**Guardrails:** <PII, injection, SLA>  
**Interfaces:** `retrieve(query)`, `rerank(docs)`, `open_ticket(json)`  
**Demo Path:** <3 steps for 90-sec demo>  
**Non-Goals:** <parked nice-to-haves>  
**Rollback:** <how to disable quickly>

## Eval Item YAML (example)
```yaml
- id: sop-001
  type: lookup
  query: "Torque spec for motor M3 on Alpha v2"
  expected_citations:
    - "docs/sop/alpha-v2/mech/torque.md#M3"
  assertions:
    - must_contain: "Nm"
    - must_cite_expected: true
    - must_not_hallucinate: true
