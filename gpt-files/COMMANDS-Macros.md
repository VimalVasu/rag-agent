# Commands & Macros

- `@scope` — Shrink current request into ≤2 tasks (core/support) with acceptance tests + Friday demo script.
- `@triage` — Sort backlog into Now / Next / Later; justify Now by metric impact.
- `@evalseed` — Generate 10 eval items (lookup/procedural/escalate) as YAML for today’s feature.
- `@numbers` — Add before/after metric targets + measurement plan (log P@5, p95, cost).
- `@tracecheck` — Define what to inspect in traces + 1 guardrail (jailbreak/PII) for this week.
- `@dailystart` — Run the 35/90/20/10 loop with checklists and timers.
- `@friday` — Compile demo script, metric diffs, 3 learnings, 1 bottleneck for next week.
- `@rescope` — Task exceeded 2 days → split into smaller stories with DoD per story.
- `@jobpack` — Format this week’s artifacts into an interview-ready bundle (README bullets + links + talking points).
-  `@claudeprompt` — Create a ready-to-paste Claude Code prompt to implement a PRD slice (files + tests + run steps). Input: your PRD/context + constraints. Output: a single prompt that asks Claude Code to scaffold code, add pytest tests, and run them by default.
    Usage:
@claudeprompt
CONTEXT
<Brief PRD slice or goal: what to build, why it matters, target interfaces and metrics.>
SCOPE
<Files to create or modify; languages; must-have functions/APIs; CLI endpoints.>
CONSTRAINTS
<Runtime/version, allowed deps, no-network if needed, performance/latency budget.>

Produce a single Claude Code prompt that instructs it to:
1) Create the specified files and folders with production-ready, readable code.
2) Include unit/integration tests using pytest (or language-idiomatic test framework).
3) Add a CLI or script entry points to run the code.
4) Provide commands to set up env, run tests, and run a demo scenario.
5) Keep outputs deterministic and instrumented (timings/logs as applicable).

Prompt structure to generate (Claude should receive exactly this):

---BEGIN PROMPT FOR CLAUDE CODE---
You are Claude Code. Implement the following PRD slice.

CONTEXT
<inject CONTEXT above>

FILES & TASKS
- Create/modify:
  <file list with short bullet specs, e.g. src/ingest.py, src/retrieve.py ...>
- Implement key interfaces:
  <e.g., retrieve(query)->docs, rerank(docs)->docs, open_ticket(json)->draft>
- Determinism & logging:
  <hashing scheme, timestamps, basic tracing/log output>

TESTS (must write and run)
- Framework: pytest
- Write tests that verify:
  - Happy path behavior of each interface
  - Edge cases (empty inputs, unicode, large input)
  - Deterministic IDs/ordering where applicable
  - Performance smoke: single-call wall time under X seconds (mark as slow if needed)

RUN INSTRUCTIONS (must print in final message)
- Env setup commands (venv + pip install or language equivalent)
- Test command: `pytest -q`
- Demo command(s): `python -m <entrypoint>` with example args

CONSTRAINTS
<inject CONSTRAINTS above>

DELIVERABLES
- All source files
- All test files
- A short README section or printed summary with how to run tests & demo

Quality bar:
- Clear comments on tricky parts; small, composable functions; no dead code.
---END PROMPT FOR CLAUDE CODE---
