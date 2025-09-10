"""
Lightweight cross-encoder reranker for document candidates.

This module provides a fast, CPU-friendly reranking system that uses a small
cross-encoder model to reorder retrieved document candidates. It combines
cross-encoder scores with existing retrieval scores using a weighted blend.

Key features:
- Batched scoring for efficiency
- Fallback to fused scores if model loading fails
- Deterministic output with proper normalization
- CLI interface for testing
- Built-in profiling and self-test capabilities
"""

import argparse
import dataclasses
import json
import math
import os
import random
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclasses.dataclass
class Candidate:
    """Document candidate for reranking."""
    id: str
    title: str
    text: str
    fused_score: float
    bm25_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    # Fields added by reranker:
    ce_score: float = 0.0
    ce_prob: float = 0.5
    final_score: float = 0.0
    snippet: str = ""


class CrossEncoderModel:
    """Wrapper for cross-encoder model with fallback."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model with fallback."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            print(f"Loaded cross-encoder model: {self.model_name}")
        except ImportError:
            print("Warning: sentence-transformers not available, using fallback scorer")
            self.model = None
        except Exception as e:
            print(f"Warning: Failed to load model {self.model_name}: {e}")
            print("Using fallback scorer")
            self.model = None
    
    def predict(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        """Score query-document pairs."""
        if self.model is None:
            # Fallback: return neutral scores
            return [0.0] * len(pairs)
        
        try:
            scores = self.model.predict(pairs, batch_size=batch_size)
            return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        except Exception as e:
            print(f"Warning: Model prediction failed: {e}")
            return [0.0] * len(pairs)


# Global model cache
_MODEL_CACHE: Dict[str, CrossEncoderModel] = {}


def load_model(model_name: Optional[str] = None) -> CrossEncoderModel:
    """Load and cache cross-encoder model."""
    if model_name is None:
        model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = CrossEncoderModel(model_name)
    
    return _MODEL_CACHE[model_name]


def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def score_pairs(model: CrossEncoderModel, pairs: List[Tuple[str, str]], 
                batch_size: int = 16) -> List[float]:
    """Score query-document pairs in batches."""
    if not pairs:
        return []
    
    all_scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_scores = model.predict(batch, batch_size=len(batch))
        all_scores.extend(batch_scores)
    
    return all_scores


def make_snippet(text: str, max_chars: int = 220) -> str:
    """Create a snippet from text."""
    if not text:
        return ""
    
    # Clean text
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    
    if len(text) <= max_chars:
        return text
    
    # Try to break at sentence boundary
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    last_exclaim = truncated.rfind('!')
    last_question = truncated.rfind('?')
    
    sentence_end = max(last_period, last_exclaim, last_question)
    
    if sentence_end > max_chars * 0.6:  # If we found a reasonable sentence break
        return text[:sentence_end + 1].strip()
    
    # Otherwise truncate and add ellipsis
    return truncated.rstrip() + "..."


def minmax_normalize(scores: List[float]) -> List[float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.5] * len(scores)
    
    return [(score - min_score) / (max_score - min_score) for score in scores]


def rerank(query: str, 
           candidates: List[Candidate], 
           top_k: int = 5, 
           alpha: float = 0.8,
           model_name: Optional[str] = None) -> List[Candidate]:
    """
    Rerank candidates using cross-encoder scores.
    
    Args:
        query: Search query
        candidates: List of candidates to rerank
        top_k: Number of top results to return
        alpha: Weight for cross-encoder score vs fused score [0, 1]
        model_name: Cross-encoder model name (optional)
    
    Returns:
        Top-k candidates sorted by final_score descending
    """
    # Set seeds for determinism
    set_seeds()
    
    # Handle empty input
    if not candidates:
        return []
    
    # Limit top_k to available candidates
    top_k = min(top_k, len(candidates))
    
    # Load model
    model = load_model(model_name)
    
    # Get batch size from environment
    batch_size = int(os.getenv("RERANK_BATCH", "16"))
    
    # Build query-document pairs
    pairs = [(query, candidate.text) for candidate in candidates]
    
    # Score pairs
    ce_scores = score_pairs(model, pairs, batch_size=batch_size)
    
    # Convert to probabilities using logistic function
    ce_probs = [1.0 / (1.0 + math.exp(-score)) for score in ce_scores]
    
    # Normalize fused scores
    fused_scores = [c.fused_score for c in candidates]
    fused_normalized = minmax_normalize(fused_scores)
    
    # Calculate final scores and update candidates
    for i, candidate in enumerate(candidates):
        candidate.ce_score = ce_scores[i]
        candidate.ce_prob = ce_probs[i]
        candidate.final_score = alpha * ce_probs[i] + (1 - alpha) * fused_normalized[i]
        candidate.snippet = make_snippet(candidate.text)
    
    # Sort by final score descending, then by id for stability
    candidates.sort(key=lambda x: (-x.final_score, x.id))
    
    return candidates[:top_k]


def profile_rerank(query: str, candidates: List[Candidate]) -> Dict[str, Any]:
    """Profile reranking performance."""
    start_time = time.time()
    
    # Run reranking
    rerank(query, candidates)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    n_candidates = len(candidates)
    time_per_doc = (total_time / n_candidates * 1000) if n_candidates > 0 else 0
    
    batch_size = int(os.getenv("RERANK_BATCH", "16"))
    
    return {
        "n": n_candidates,
        "batch_size": batch_size,
        "t_total_s": round(total_time, 4),
        "t_per_doc_ms": round(time_per_doc, 2)
    }


def create_test_candidates() -> List[Candidate]:
    """Create synthetic test candidates."""
    candidates = [
        Candidate(
            id="c1",
            title="Motor M3 Specifications",
            text="The torque specification for motor M3 is 12 Nm. This motor requires careful calibration and maintenance.",
            fused_score=0.85,
            bm25_rank=1,
            dense_rank=2
        ),
        Candidate(
            id="c2", 
            title="Sensor Calibration Guide",
            text="Calibrate sensor S2 with offset 0.3. This ensures accurate readings during operation.",
            fused_score=0.62,
            bm25_rank=3,
            dense_rank=1
        ),
        Candidate(
            id="c3",
            title="Safety Protocols",
            text="Always wear safety gear when working with motors and sensors. Follow proper shutdown procedures.",
            fused_score=0.41,
            bm25_rank=5,
            dense_rank=4
        ),
        Candidate(
            id="c4",
            title="M3 Motor Installation",
            text="Motor M3 installation requires specific torque values. Use proper tools and follow the manual.",
            fused_score=0.73,
            bm25_rank=2,
            dense_rank=3
        ),
        Candidate(
            id="c5",
            title="General Maintenance",
            text="Regular maintenance schedules help prevent equipment failure. Document all procedures.",
            fused_score=0.28,
            bm25_rank=4,
            dense_rank=5
        )
    ]
    return candidates


def run_selftest():
    """Run self-test with synthetic data."""
    print("Running self-test...")
    
    query = "Torque spec for motor M3"
    candidates = create_test_candidates()
    
    print(f"Query: {query}")
    print(f"Input candidates: {len(candidates)}")
    
    # Profile the reranking
    profile_data = profile_rerank(query, candidates.copy())
    print(f"Profile: {profile_data}")
    
    # Rerank candidates
    results = rerank(query, candidates, top_k=3, alpha=0.8)
    
    print(f"\nTop-3 Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.id} (final_score: {result.final_score:.3f})")
        print(f"   Title: {result.title}")
        print(f"   CE Score: {result.ce_score:.3f}, CE Prob: {result.ce_prob:.3f}")
        print(f"   Snippet: {result.snippet}")
        print()
    
    # Verify that a relevant candidate ranks highly
    top_result = results[0]
    if "torque" in top_result.text.lower() and "m3" in top_result.text.lower():
        print("PASS: Self-test passed: Relevant candidate ranked #1")
    else:
        print("WARN: Self-test warning: Most relevant candidate not #1")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Rerank document candidates using cross-encoder")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--candidates", type=str, help="Path to candidates JSON file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results")
    parser.add_argument("--alpha", type=float, default=0.8, help="Weight for CE vs fused score")
    parser.add_argument("--model", type=str, help="Cross-encoder model name")
    parser.add_argument("--selftest", action="store_true", help="Run self-test")
    parser.add_argument("--profile", action="store_true", help="Show profiling info")
    
    args = parser.parse_args()
    
    if args.selftest:
        run_selftest()
        return
    
    if not args.query:
        print("Error: --query required (or use --selftest)")
        return
    
    if not args.candidates:
        print("Error: --candidates path required (or use --selftest)")
        return
    
    # Load candidates from JSON file
    try:
        with open(args.candidates, 'r', encoding='utf-8') as f:
            candidates_data = json.load(f)
        
        candidates = []
        for data in candidates_data:
            candidate = Candidate(
                id=data["id"],
                title=data["title"], 
                text=data["text"],
                fused_score=data["fused_score"],
                bm25_rank=data.get("bm25_rank"),
                dense_rank=data.get("dense_rank")
            )
            candidates.append(candidate)
    
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error loading candidates: {e}")
        return
    
    # Profile if requested
    if args.profile:
        profile_data = profile_rerank(args.query, candidates.copy())
        print(f"Profile: {json.dumps(profile_data, indent=2)}")
    
    # Rerank candidates
    results = rerank(
        query=args.query,
        candidates=candidates,
        top_k=args.top_k,
        alpha=args.alpha,
        model_name=args.model
    )
    
    # Convert to JSON-serializable format
    output = []
    for result in results:
        output.append({
            "id": result.id,
            "title": result.title,
            "text": result.text,
            "fused_score": result.fused_score,
            "bm25_rank": result.bm25_rank,
            "dense_rank": result.dense_rank,
            "ce_score": result.ce_score,
            "ce_prob": result.ce_prob,
            "final_score": result.final_score,
            "snippet": result.snippet
        })
    
    # Output JSON
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()