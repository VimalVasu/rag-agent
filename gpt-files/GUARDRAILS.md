# Guardrails

Scope & WIP
- Never >2 concurrent initiatives; use `@triage` to force trade-offs.

Planning Discipline
- Planning >35m → freeze and build happy path
- Any story >2 days → `@rescope` into smaller units

Security & Safety
- Prompt-injection: sanitize pasted URLs/code blocks; restrict tool-allowed instructions
- PII: redact logs/traces
- External writes (tickets/orders): human-in-the-loop approval

SLA Awareness
- Maintain p95 budget and fail CI on breach
- Record trade-offs explicitly in Changelog
