"""
Minimal, deterministic, markdown-aware ingester for a RAG system.
"""

import argparse
import json
import re
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class Document:
    """Represents a loaded document."""
    path: str
    title: str
    text: str
    doc_id: str


@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    chunk_id: str
    doc_id: str
    path: str
    title: str
    headings: List[str]
    text: str
    start_char: int
    end_char: int
    token_estimate: int
    created_at: str
    sha1: str


def load_documents(doc_dir: str, exts=(".md", ".txt")) -> List[Document]:
    """Load documents from directory with specified extensions."""
    documents = []
    doc_path = Path(doc_dir)
    
    if not doc_path.exists():
        return documents
    
    # Get all files with specified extensions, sorted for deterministic ordering
    files = []
    for ext in exts:
        files.extend(doc_path.glob(f"**/*{ext}"))
    files.sort()
    
    for file_path in files:
        try:
            # Read with UTF-8 and replace errors
            text = file_path.read_text(encoding="utf-8", errors="replace")
            
            # Normalize text (strip and normalize whitespace)
            normalized_text = re.sub(r'\s+', ' ', text.strip())
            
            # Extract title: first H1 or filename stem
            title = None
            h1_match = re.search(r'^#\s+(.+)', text, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1).strip()
            else:
                title = file_path.stem
            
            # Generate deterministic doc_id from normalized full text
            doc_id = hashlib.sha1(normalized_text.encode('utf-8')).hexdigest()
            
            documents.append(Document(
                path=str(file_path),
                title=title,
                text=text,
                doc_id=doc_id
            ))
            
        except Exception:
            # Skip files that can't be read
            continue
    
    return documents


def split_markdown(document: Document, max_chars=900, overlap=150, min_chars=200) -> List[Chunk]:
    """Split document into chunks with markdown-aware boundaries."""
    chunks = []
    text = document.text
    
    # Skip documents that are too short after normalization
    normalized_text = re.sub(r'\s+', ' ', text.strip())
    if len(normalized_text) < min_chars:
        return chunks
    
    # Find all headings and code fence boundaries
    heading_matches = list(re.finditer(r'^(#{1,3})\s+(.+)', text, re.MULTILINE))
    code_fence_matches = list(re.finditer(r'^```', text, re.MULTILINE))
    
    # Build code fence ranges (start, end) pairs
    code_fence_ranges = []
    for i in range(0, len(code_fence_matches), 2):
        if i + 1 < len(code_fence_matches):
            start = code_fence_matches[i].start()
            end = code_fence_matches[i + 1].end()
            code_fence_ranges.append((start, end))
    
    def is_inside_code_fence(pos: int) -> bool:
        """Check if position is inside a code fence."""
        for start, end in code_fence_ranges:
            if start <= pos <= end:
                return True
        return False
    
    # Create sections based on headings
    sections = []
    
    if not heading_matches:
        # No headings, treat entire document as one section
        sections.append({
            'start': 0,
            'end': len(text),
            'headings': [],
            'text': text
        })
    else:
        # Process sections between headings
        for i, match in enumerate(heading_matches):
            start = match.start()
            end = len(text) if i == len(heading_matches) - 1 else heading_matches[i + 1].start()
            
            # Build heading stack up to current level
            level = len(match.group(1))
            heading = match.group(2).strip()
            
            # Get parent headings
            parent_headings = []
            for j in range(i - 1, -1, -1):
                prev_match = heading_matches[j]
                prev_level = len(prev_match.group(1))
                if prev_level < level:
                    parent_headings.insert(0, prev_match.group(2).strip())
                    level = prev_level
            
            parent_headings.append(heading)
            
            sections.append({
                'start': start,
                'end': end,
                'headings': parent_headings,
                'text': text[start:end]
            })
    
    # Further split long sections by characters
    chunk_index = 0
    created_at = datetime.utcnow().isoformat() + 'Z'
    
    for section in sections:
        section_text = section['text']
        section_start = section['start']
        
        if len(section_text) <= max_chars:
            # Section fits in one chunk
            chunk_text = section_text.strip()
            if chunk_text:
                token_estimate = len(chunk_text.split())
                chunk_sha1 = hashlib.sha1(chunk_text.encode('utf-8')).hexdigest()
                
                chunks.append(Chunk(
                    chunk_id=f"{document.doc_id}:{chunk_index}",
                    doc_id=document.doc_id,
                    path=document.path,
                    title=document.title,
                    headings=section['headings'],
                    text=chunk_text,
                    start_char=section_start,
                    end_char=section_start + len(section_text),
                    token_estimate=token_estimate,
                    created_at=created_at,
                    sha1=chunk_sha1
                ))
                chunk_index += 1
        else:
            # Split long section into smaller chunks
            pos = 0
            while pos < len(section_text):
                # Find chunk boundaries
                chunk_end = min(pos + max_chars, len(section_text))
                
                # Don't split inside code fences
                absolute_pos = section_start + pos
                absolute_end = section_start + chunk_end
                
                # Check if we're ending inside a code fence
                if is_inside_code_fence(absolute_end) and chunk_end < len(section_text):
                    # Find the end of the code fence
                    for fence_start, fence_end in code_fence_ranges:
                        if fence_start <= absolute_end <= fence_end:
                            chunk_end = min(fence_end - section_start, len(section_text))
                            break
                
                chunk_text = section_text[pos:chunk_end].strip()
                if chunk_text:
                    token_estimate = len(chunk_text.split())
                    chunk_sha1 = hashlib.sha1(chunk_text.encode('utf-8')).hexdigest()
                    
                    chunks.append(Chunk(
                        chunk_id=f"{document.doc_id}:{chunk_index}",
                        doc_id=document.doc_id,
                        path=document.path,
                        title=document.title,
                        headings=section['headings'],
                        text=chunk_text,
                        start_char=section_start + pos,
                        end_char=section_start + chunk_end,
                        token_estimate=token_estimate,
                        created_at=created_at,
                        sha1=chunk_sha1
                    ))
                    chunk_index += 1
                
                # Move position forward with overlap
                if chunk_end >= len(section_text):
                    break
                pos = max(pos + 1, chunk_end - overlap)
    
    return chunks


def build_corpus(doc_dir: str, max_chars=900, overlap=150, min_chars=200) -> List[Chunk]:
    """Build complete corpus by loading documents and splitting them into chunks."""
    documents = load_documents(doc_dir)
    chunks = []
    
    for document in documents:
        doc_chunks = split_markdown(document, max_chars, overlap, min_chars)
        chunks.extend(doc_chunks)
    
    return chunks


def save_jsonl(chunks: List[Chunk], out_path: str) -> None:
    """Save chunks to JSONL file (one JSON object per line)."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            json.dump(asdict(chunk), f, ensure_ascii=False)
            f.write('\n')


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest documents for RAG system")
    parser.add_argument("--doc_dir", required=True, help="Directory containing documents")
    parser.add_argument("--out", required=True, help="Output JSONL file path")
    parser.add_argument("--max_chars", type=int, default=900, help="Maximum characters per chunk")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap between chunks")
    parser.add_argument("--min_chars", type=int, default=200, help="Minimum characters for document inclusion")
    
    args = parser.parse_args()
    
    # Load all documents
    documents = load_documents(args.doc_dir)
    
    # Build corpus
    chunks = build_corpus(args.doc_dir, args.max_chars, args.overlap, args.min_chars)
    
    # Count skipped short files
    total_files = len(documents)
    files_with_chunks = len(set(chunk.path for chunk in chunks))
    skipped_short = total_files - files_with_chunks
    
    # Save corpus
    save_jsonl(chunks, args.out)
    
    # Print summary
    print(f"files={total_files}, chunks={len(chunks)}, skipped_short={skipped_short}")


if __name__ == "__main__":
    main()