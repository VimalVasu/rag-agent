# Metrics Cheat-Sheet

Primary
- Retrieval quality: **P@5** (primary), nDCG@10 (secondary)
- Grounding: **citation supports claim** rate
- Latency: **p50/p95** end-to-end
- Cost: **$/query (avg)**
- Vision: **top-1/top-3** accuracy on labeled photos

Measurement Notes
- Log dataset version + seed
- Store eval results in `/evals/results.json` with timestamps
- Track trend weekly; add simple threshold alerts (CI fails if P@5 −3pts or p95 > budget)
