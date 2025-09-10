"""
Hybrid retrieval module combining BM25 keyword search and semantic embeddings search.
Uses Reciprocal Rank Fusion (RRF) to combine rankings from both approaches.
"""

import json
import re
import unicodedata
from typing import Dict, List, Optional, Any, Union
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase alphanumeric tokens, unicode-safe.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of normalized tokens
    """
    if not text:
        return []
    
    # Normalize unicode characters
    normalized = unicodedata.normalize('NFKC', text)
    
    # Extract alphanumeric tokens, convert to lowercase
    tokens = re.findall(r'\w+', normalized.lower())
    
    return tokens


def rrf(ranks: Dict[str, List[int]], k: int = 60) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion: combines multiple rankings using RRF formula.
    
    Args:
        ranks: Dictionary mapping item_id to list of ranks (0-indexed)
        k: RRF parameter (default 60)
        
    Returns:
        Dictionary mapping item_id to RRF score
    """
    scores = {}
    
    for item_id, rank_list in ranks.items():
        score = 0.0
        for rank in rank_list:
            score += 1.0 / (k + rank + 1)  # rank is 0-indexed, so add 1
        scores[item_id] = score
    
    return scores


def make_snippet(text: str, query: str, max_len: int = 240) -> str:
    """
    Create a snippet centered around the first query match, or fallback to head.
    
    Args:
        text: Source text
        query: Query string to highlight
        max_len: Maximum snippet length
        
    Returns:
        Snippet string
    """
    if not text:
        return ""
    
    if len(text) <= max_len:
        return text
    
    # Try to find first query token in text
    query_tokens = tokenize(query)
    text_lower = text.lower()
    
    best_pos = -1
    for token in query_tokens:
        pos = text_lower.find(token.lower())
        if pos != -1:
            best_pos = pos
            break
    
    if best_pos == -1:
        # No match found, return head (account for ellipsis)
        if max_len > 3:
            return text[:max_len-3].rstrip() + "..."
        else:
            return text[:max_len]
    
    # Reserve space for ellipsis
    ellipsis_space = 3  # "..."
    available_len = max_len
    
    # Center around the match
    start = max(0, best_pos - available_len // 2)
    end = start + available_len
    
    if end > len(text):
        end = len(text)
        start = max(0, end - available_len)
    
    snippet = text[start:end]
    
    # Add ellipsis if we're not at the boundaries
    needs_start_ellipsis = start > 0
    needs_end_ellipsis = end < len(text)
    
    if needs_start_ellipsis and needs_end_ellipsis:
        # Need both ellipsis - trim content to make room
        content_len = max_len - 6  # "..." + content + "..."
        if content_len > 0:
            snippet = text[start:start+content_len]
            snippet = "..." + snippet + "..."
        else:
            snippet = "..."
    elif needs_start_ellipsis:
        # Need start ellipsis only
        content_len = max_len - 3
        if content_len > 0:
            snippet = text[start:start+content_len]
            snippet = "..." + snippet
        else:
            snippet = "..."
    elif needs_end_ellipsis:
        # Need end ellipsis only  
        content_len = max_len - 3
        if content_len > 0:
            snippet = text[start:start+content_len]
            snippet = snippet + "..."
        else:
            snippet = "..."
    
    return snippet


class HybridRetriever:
    """
    Hybrid retrieval system combining BM25 keyword search and semantic embeddings.
    """
    
    def __init__(self, chunks: List[Dict[str, Any]], embedder=None, text_key: str = "text", title_key: str = "title"):
        """
        Initialize retriever with chunks and embedder.
        
        Args:
            chunks: List of document chunks with metadata
            embedder: Object with .encode(list[str]) -> np.ndarray method
            text_key: Key for text content in chunks
            title_key: Key for title in chunks
        """
        self.chunks = chunks
        self.text_key = text_key
        self.title_key = title_key
        self.embedder = embedder
        
        # If no embedder provided, try to use sentence-transformers default
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                # Fallback to None - will skip embedding search
                pass
        
        # Build BM25 index
        texts = [chunk.get(text_key, "") for chunk in chunks]
        tokenized_texts = [tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        # Build embedding index if embedder available
        self.embeddings = None
        if self.embedder is not None:
            try:
                self.embeddings = self.embedder.encode(texts)
                if isinstance(self.embeddings, list):
                    self.embeddings = np.array(self.embeddings)
            except Exception:
                # If encoding fails, skip embeddings
                self.embeddings = None
    
    @classmethod
    def from_chunks(cls, chunks: List[Dict[str, Any]], *, embedder=None, text_key: str = "text", title_key: str = "title") -> 'HybridRetriever':
        """
        Create retriever from list of chunks.
        
        Args:
            chunks: List of document chunks
            embedder: Embedder object
            text_key: Key for text content
            title_key: Key for title
            
        Returns:
            HybridRetriever instance
        """
        return cls(chunks, embedder=embedder, text_key=text_key, title_key=title_key)
    
    @classmethod
    def from_jsonl(cls, path: str, **kwargs) -> 'HybridRetriever':
        """
        Create retriever from JSONL file.
        
        Args:
            path: Path to JSONL file
            **kwargs: Additional arguments for from_chunks
            
        Returns:
            HybridRetriever instance
        """
        chunks = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        
        return cls.from_chunks(chunks, **kwargs)
    
    def search(self, query: str, *, k_bm25: int = 50, k_embed: int = 50, k: int = 20) -> List[Dict[str, Any]]:
        """
        Search using hybrid BM25 + embedding approach with RRF fusion.
        
        Args:
            query: Search query
            k_bm25: Top-K results from BM25
            k_embed: Top-K results from embeddings  
            k: Final top-K results to return
            
        Returns:
            List of results with metadata and scores
        """
        # Handle empty or whitespace query
        if not query or not query.strip():
            return []
        
        query = query.strip()
        query_tokens = tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores and ranks
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_top_k = bm25_ranked[:k_bm25]
        
        # Get embedding similarity scores and ranks  
        embed_top_k = []
        if self.embeddings is not None and self.embedder is not None:
            try:
                query_embedding = self.embedder.encode([query])
                if isinstance(query_embedding, list):
                    query_embedding = np.array(query_embedding)
                
                # Compute cosine similarity
                query_norm = np.linalg.norm(query_embedding)
                doc_norms = np.linalg.norm(self.embeddings, axis=1)
                
                # Avoid division by zero
                similarities = np.zeros(len(self.embeddings))
                valid_mask = (query_norm > 1e-8) & (doc_norms > 1e-8)
                
                if valid_mask.any():
                    similarities[valid_mask] = np.dot(self.embeddings[valid_mask], query_embedding.T).flatten() / (doc_norms[valid_mask] * query_norm)
                
                embed_ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
                embed_top_k = embed_ranked[:k_embed]
            except Exception:
                # If embedding search fails, continue with BM25 only
                pass
        
        # Collect all candidate items and their ranks
        ranks_dict = {}
        
        # Add BM25 ranks
        for rank, (doc_idx, score) in enumerate(bm25_top_k):
            chunk_id = self.chunks[doc_idx].get('chunk_id', str(doc_idx))
            if chunk_id not in ranks_dict:
                ranks_dict[chunk_id] = []
            ranks_dict[chunk_id].append(rank)
        
        # Add embedding ranks
        for rank, (doc_idx, score) in enumerate(embed_top_k):
            chunk_id = self.chunks[doc_idx].get('chunk_id', str(doc_idx))
            if chunk_id not in ranks_dict:
                ranks_dict[chunk_id] = []
            ranks_dict[chunk_id].append(rank)
        
        # Apply RRF fusion
        rrf_scores = rrf(ranks_dict, k=60)
        
        # Build result objects
        results = []
        for chunk_id, rrf_score in rrf_scores.items():
            # Find the chunk by chunk_id
            doc_idx = None
            for idx, chunk in enumerate(self.chunks):
                if chunk.get('chunk_id', str(idx)) == chunk_id:
                    doc_idx = idx
                    break
            
            if doc_idx is None:
                continue
            
            chunk = self.chunks[doc_idx]
            text = chunk.get(self.text_key, "")
            
            # Get original ranks
            bm25_rank = None
            embed_rank = None
            
            for rank, (idx, _) in enumerate(bm25_top_k):
                if idx == doc_idx:
                    bm25_rank = rank + 1  # Convert to 1-indexed
                    break
            
            for rank, (idx, _) in enumerate(embed_top_k):
                if idx == doc_idx:
                    embed_rank = rank + 1  # Convert to 1-indexed
                    break
            
            result = {
                'chunk_id': chunk.get('chunk_id', str(doc_idx)),
                'doc_id': chunk.get('doc_id', ''),
                'path': chunk.get('path', ''),
                'title': chunk.get(self.title_key, ''),
                'snippet': make_snippet(text, query),
                'bm25_rank': bm25_rank,
                'embed_rank': embed_rank,
                'rrf_score': rrf_score
            }
            results.append(result)
        
        # Sort by RRF score descending, then by chunk_id ascending for stability
        results.sort(key=lambda x: (-x['rrf_score'], str(x['chunk_id'])))
        
        return results[:k]
    
    def close(self):
        """Optional cleanup method (no-op)."""
        pass