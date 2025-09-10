You are Claude Code. Implement a minimal, deterministic, markdown-aware ingester for a RAG system.

TASKS
1) In file `src/ingest.py` write the code for:
   - dataclasses: Document, Chunk (fields below).
   - functions:
     * load_documents(doc_dir: str, exts=(".md",".txt")) -> list[Document]
     * split_markdown(document: Document, max_chars=900, overlap=150, min_chars=200) -> list[Chunk]
     * build_corpus(doc_dir, max_chars=900, overlap=150, min_chars=200) -> list[Chunk]
     * save_jsonl(chunks, out_path)
     * main() CLI: --doc_dir, --out, --max_chars, --overlap, --min_chars
   - Behavior:
     * Read UTF-8 with errors="replace"; skip files with < MIN_CHARS after normalization.
     * Title = first H1 (“# ”) else filename stem.
     * Markdown-aware split:
       - Prefer boundaries at H2/H3 (##/###).
       - Never split inside triple-backtick code fences; treat fenced blocks atomically (may exceed max_chars).
       - After heading splitting, further split long sections by characters to max_chars with overlap.
     * Deterministic ordering (sort by path).
     * IDs: doc_id = sha1(normalized full text); chunk_id = f"{doc_id}:{i}" per order; sha1 = sha1(chunk text).
     * Fields per Chunk:
       chunk_id, doc_id, path, title, headings(list[str]), text, start_char, end_char,
       token_estimate(int via word count), created_at(UTC ISO8601), sha1.
     * save_jsonl writes one JSON object per line.
     * main() prints summary: files=?, chunks=?, skipped_short=?.

2) Create tests in `tests/test_ingest.py` (pytest) that verify:
   - loads .md and .txt; title extraction prefers H1; filename fallback works.
   - heading-based splitting with code fences is respected (no splits inside ``` blocks).
   - length bounds and ~overlap are enforced (tolerate ±20 chars).
   - offsets cover document regions; deterministic IDs and hashes are stable.
   - short files skipped; non-ASCII preserved; JSONL schema keys/types present.
   - CLI runs and prints a one-line summary.

CONSTRAINTS
- Python 3.10+, standard library only (dataclasses, re, json, hashlib, pathlib, datetime, argparse).
- No network. No heavy markdown libraries.
- Keep logic simple and readable; comment tricky parts (code-fence state machine, heading stack).

DELIVERABLES
- `src/ingest.py` with CLI.
- `tests/test_ingest.py`.
- Running `pytest -q` should pass all tests.
- Example run:
  python -m src.ingest --doc_dir data/docs --out data/corpus.jsonl --max_chars 900 --overlap 150 --min_chars 200
