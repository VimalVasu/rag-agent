Today: Yes—this is mostly a coding day: build retrieve(), add a simple rerank(), wire a dry-run open_ticket(), and run a 15-query test.

Risky: Model downloads & env setup can eat time—use light models and keep everything in-memory.

One number to move: P@5 ≥ 70% on today’s 15-query slice.

What you’ll actually do (clear + concrete)
1) Where you’ll code (repo layout)

Create a tiny repo so everything has a home:

rag-agent/
├─ data/
│  ├─ docs/                # your SOPs/notes as .md/.pdf (start with ~20 md files)
│  └─ queries/dev.yaml     # 15 queries (lookup/procedural/escalate)
├─ src/
│  ├─ ingest.py            # loads docs → text chunks
│  ├─ retrieve.py          # hybrid search: BM25 + embeddings
│  ├─ rerank.py            # reorders top candidates → better top-5
│  ├─ ticket_tool.py       # open_ticket(json) → draft (no external write)
│  ├─ pipeline.py          # glue: retrieve → rerank → (maybe) ticket
│  └─ evals.py             # runs P@5, nDCG@10, p50/p95 timing
├─ ops/metrics.md          # jot p95 & $/query and today’s P@5
├─ requirements.txt
└─ README.md

2) What data you’ll use (and where it comes from)

Docs (data/docs/): drop your existing SOPs/notes/FAQs as Markdown. If you don’t have them handy, create 10–20 small .md files with realistic titles and headings—enough to test search quality.

Queries (data/queries/dev.yaml): write 15 short questions (5 lookup, 5 procedural, 5 escalate). Example:

- id: sop-lookup-001
  type: lookup
  query: "Torque spec for motor M3 on Alpha v2"

- id: sop-procedural-001
  type: procedural
  query: "How do I calibrate sensor S2 on Beta v1?"

- id: escalate-001
  type: escalate
  query: "Field unit Z9 motor seized; need repair ticket"

3) How you’ll build it (fast path)
A. Set up (10–15m)

requirements.txt (keep it light):

rank-bm25
sentence-transformers
faiss-cpu      # optional; you can skip and do cosine manually with numpy
numpy
scikit-learn
pydantic


Then:

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

B. Ingest docs (10m) — src/ingest.py

Walk data/docs/, read .md, split by headings or ~500–800 chars.

Output a list of chunks: [{id, doc_id, text, title}].

C. Hybrid retrieve (35–45m) — src/retrieve.py

BM25: use rank_bm25 on tokenized chunks → get top 50.

Embeddings: use a small SentenceTransformer (e.g., all-MiniLM-L6-v2) to embed chunks and the query → cosine to get top 50.

Fuse: combine via Reciprocal Rank Fusion (RRF) for simplicity:

take BM25 top-k ranks and Embedding top-k ranks, compute score = Σ 1/(k + rank); return fused top-20.

Return: List[{chunk_id, title, snippet, fused_score}].

Tiny example (core logic only):

# retrieve.py
from rank_bm25 import BM25Okapi
import numpy as np

def rrf(ranks, k=60):
    # ranks: dict[chunk_id] -> list_of_ranks_from_each_system
    scores = {}
    for cid, rr in ranks.items():
        scores[cid] = sum(1.0/(k + r) for r in rr)
    return scores

# Tip: keep embeddings in memory: np.ndarray [num_chunks, dim]

D. Rerank (20–25m) — src/rerank.py

Take the fused top-20, build pairs (query, chunk_text), and score with a small cross-encoder or an LLM scoring prompt if you prefer.

Return the top-5 with titles and 1–2 sentence snippets (these are your “P@5” candidates).

E. Glue + cache (10m) — src/pipeline.py

answer(query):

retrieve(query) → top20

rerank(top20) → top5

package: {"top5": [...], "citations": [...], "trace": {timings}}

Add a 10-minute in-memory dict: {hash(query): result, ttl} to keep p95 low on repeats.

F. Ticket tool (10–15m) — src/ticket_tool.py

open_ticket(payload: dict) -> dict that returns a draft only (no external call).

When the query looks like an escalation (contains “seized”, “down”, “RMA”, etc.), call the tool with fields {title, severity, unit_id, summary} and attach the suggested top citation.

Example shape:

def open_ticket(json_in: dict) -> dict:
    return {
        "title": json_in.get("title",""),
        "severity": json_in.get("severity","P3"),
        "unit_id": json_in.get("unit_id","unknown"),
        "summary": json_in.get("summary",""),
        "status": "DRAFT"
    }

4) How you’ll evaluate it (and write the results)
Quick eval runner — src/evals.py (20m total with instrumentation)

Loop over the 15 queries:

Run answer(query), measure start→end time.

P@5: mark 1 if the correct doc appears in the top-5 (you can label “expected doc id” in your YAML for today).

Track p50/p95 latency and count tokens (or just approximate cost today).

Write metrics to ops/metrics.md:

## 2025-09-08
P@5: 0.72 (↑ +0.06 from baseline)
nDCG@10: 0.64
Latency: p50 0.9s, p95 1.5s
Cost/query: ~$0.00 (no LLM)  # or your actual
Notes: RRF > simple avg; cache on.

5) How you’ll run it (commands)
# 1) Ingest and warm
python -c "from src.ingest import load; chunks=load('data/docs')"

# 2) Try a single query end-to-end
python -c "from src.pipeline import answer; print(answer('Torque spec for motor M3'))"

# 3) Run today’s eval slice
python -c "from src.evals import run; run('data/queries/dev.yaml')"

What to keep in mind

Scope guard: If you’re not returning a decent top-5 today, don’t chase fancy models. Fix recall (retrieve) first.

Data first: Good titles & headings in data/docs/ make a bigger difference than another model tweak.

One demo path: Show: Query → Top-5 with citations → (if needed) Draft ticket JSON in the console.

Spicy take: A clean top-5 with citations beats a flashy UI—ship the boring core now, so everything after actually works.