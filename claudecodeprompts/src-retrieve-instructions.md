You are Claude Code. Implement the following PRD slice.

CONTEXT
We need a hybrid retrieval module for a RAG system. It must combine keyword search (BM25) and semantic search (embeddings), fuse ranks with Reciprocal Rank Fusion (RRF), and return a clean top-K list for downstream reranking/answering. Results must include metadata and a short snippet for display. Tests must not rely on downloading models; they should inject a dummy embedder.

FILES & TASKS

Create src/retrieve.py exporting:

class HybridRetriever:

from_chunks(chunks: list[dict], *, text_key="text", title_key="title") → build indices in memory.

from_jsonl(path: str) → loads chunks (one JSON object per line) then delegates to from_chunks.

search(query: str, *, k_bm25=50, k_embed=50, k=20) -> list[dict]:

tokenize query

get BM25 top k_bm25 ranks

get embedding similarities top k_embed ranks

fuse via rrf (k=60 default)

return top k unique results with:
{chunk_id, doc_id, path, title, snippet, bm25_rank, embed_rank, rrf_score}

Accept an embedder object with .encode(list[str]) -> np.ndarray; default to a small Sentence-Transformers model if available, but tests will pass a dummy.

Optional close() method (noop).

Utilities:

tokenize(text: str) -> list[str] (lowercase, basic alnum splitting, unicode-safe).

rrf(ranks: dict[str, list[int]], k:int=60) -> dict[str, float].

make_snippet(text: str, query: str, max_len: int=240) -> str (center around first match; fallback to head).

Implementation notes:

Use rank-bm25 for BM25.

Use NumPy for cosine similarity; embeddings matrix shape [N, D].

Enforce determinism: stable sorting by (rrf_score desc, chunk_id asc).

Do not crash on empty/whitespace query—return [].

TESTS (must write and run)

Create tests/test_retrieve.py using pytest with a DummyEmbedder:

.encode(texts) returns a fixed-dim vector per text deterministically (e.g., hash-based bag-of-words).

Write the following tests:

test_from_jsonl_and_search_returns_hits: tiny corpus JSONL → search returns results with required keys.

test_rrf_promotes_consensus_items: a chunk that’s rank 1 in both lists wins overall.

test_honors_k_limits: enforce k_bm25, k_embed, and final k.

test_snippet_highlights_query_terms: snippet includes a query token and ≤ 240 chars.

test_tokenize_deterministic_unicode_safe: same text → same tokens; handles non-ASCII.

test_dummy_embedder_is_deterministic: repeated calls → identical vectors (≈1e-8 tolerance).

test_empty_or_whitespace_query: returns empty list quickly.

test_rerank_ordering_stable_on_ties: tie breaks by chunk_id asc.

test_latency_smoke: ~200-chunk corpus with DummyEmbedder → search() under 0.5s (mark with @pytest.mark.slow and allow skip via env).

test_metadata_passthrough: results carry chunk_id/doc_id/path/title from input chunks.

RUN INSTRUCTIONS (must print in final message)

Env:

python -m venv .venv && source .venv/bin/activate
pip install rank-bm25 numpy sentence-transformers pytest


Tests:

pytest -q


(Optional) Demo snippet:

python - <<'PY'
import json
from src.retrieve import HybridRetriever
# Build tiny in-memory corpus:
chunks=[{"chunk_id":"d1:0","doc_id":"d1","path":"p1","title":"Motor M3","text":"Torque spec M3 is 12 Nm."},
        {"chunk_id":"d2:0","doc_id":"d2","path":"p2","title":"Calibration","text":"Calibrate sensor S2 with offset 0.3."}]
r = HybridRetriever.from_chunks(chunks, embedder=None)  # uses default if available
print(json.dumps(r.search("torque M3", k=5), indent=2))
PY


CONSTRAINTS

Python 3.10+

Dependencies: rank-bm25, numpy, sentence-transformers (runtime default embedder only; tests use dummy)

No network required for tests; avoid heavyweight downloads in CI by using DummyEmbedder there.

Deterministic outputs; readable code with comments on RRF & snippet selection.

DELIVERABLES

src/retrieve.py

tests/test_retrieve.py

Final message must include how to run tests & a tiny demo command.

Quality bar:

Clean, commented code; small functions; stable sorts; graceful handling of edge cases.