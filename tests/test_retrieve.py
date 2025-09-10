"""
Tests for the hybrid retrieval module.
Uses DummyEmbedder to avoid network dependencies and model downloads.
"""

import json
import tempfile
import os
import time
import hashlib
import numpy as np
import pytest

from src.retrieve import HybridRetriever, tokenize, rrf, make_snippet


class DummyEmbedder:
    """
    Dummy embedder that generates deterministic vectors based on text content.
    Uses a simple bag-of-words approach with hash-based features.
    """
    
    def __init__(self, dimension=384):
        self.dimension = dimension
    
    def encode(self, texts):
        """Generate deterministic embeddings for input texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Create a simple bag-of-words style embedding using hashing
            tokens = tokenize(text)
            vector = np.zeros(self.dimension)
            
            for token in tokens:
                # Use hash to map token to positions in vector
                hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
                positions = [hash_val % self.dimension, (hash_val >> 8) % self.dimension]
                for pos in positions:
                    vector[pos] += 1.0
            
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                vector = vector / norm
            
            embeddings.append(vector)
        
        return np.array(embeddings)


@pytest.fixture
def sample_chunks():
    """Sample corpus for testing."""
    return [
        {
            "chunk_id": "d1:0",
            "doc_id": "d1", 
            "path": "p1",
            "title": "Motor M3",
            "text": "Torque spec M3 is 12 Nm. The motor requires careful calibration."
        },
        {
            "chunk_id": "d2:0",
            "doc_id": "d2",
            "path": "p2", 
            "title": "Calibration",
            "text": "Calibrate sensor S2 with offset 0.3. This is critical for accuracy."
        },
        {
            "chunk_id": "d3:0",
            "doc_id": "d3",
            "path": "p3",
            "title": "Safety Protocol",
            "text": "Always wear safety gear when working with M3 motors and sensors."
        }
    ]


@pytest.fixture
def dummy_embedder():
    """Dummy embedder for testing."""
    return DummyEmbedder()


@pytest.fixture
def retriever(sample_chunks, dummy_embedder):
    """Hybrid retriever with sample data."""
    return HybridRetriever.from_chunks(sample_chunks, embedder=dummy_embedder)


def test_from_jsonl_and_search_returns_hits(tmp_path, dummy_embedder):
    """Test that JSONL loading and search returns results with required keys."""
    # Create temporary JSONL file
    jsonl_file = tmp_path / "test.jsonl"
    chunks = [
        {"chunk_id": "d1:0", "doc_id": "d1", "path": "p1", "title": "Test Doc", "text": "Python programming"},
        {"chunk_id": "d2:0", "doc_id": "d2", "path": "p2", "title": "Guide", "text": "Machine learning guide"}
    ]
    
    with open(jsonl_file, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    # Create retriever from JSONL
    retriever = HybridRetriever.from_jsonl(str(jsonl_file), embedder=dummy_embedder)
    
    # Search and verify results
    results = retriever.search("Python", k=2)
    
    assert len(results) > 0
    
    for result in results:
        # Check all required keys are present
        required_keys = ['chunk_id', 'doc_id', 'path', 'title', 'snippet', 'bm25_rank', 'embed_rank', 'rrf_score']
        for key in required_keys:
            assert key in result
        
        # Verify types
        assert isinstance(result['rrf_score'], float)
        assert result['rrf_score'] > 0


def test_rrf_promotes_consensus_items():
    """Test that RRF promotes items that rank highly in both systems."""
    # Item that's rank 1 in both lists should win
    ranks = {
        "item_A": [0, 0],  # rank 1 in both (0-indexed)
        "item_B": [1, 5],  # rank 2 and 6
        "item_C": [5, 1],  # rank 6 and 2
    }
    
    scores = rrf(ranks, k=60)
    
    # item_A should have highest score (appears first in both rankings)
    assert scores["item_A"] > scores["item_B"]
    assert scores["item_A"] > scores["item_C"]


def test_honors_k_limits(sample_chunks, dummy_embedder):
    """Test that k_bm25, k_embed, and final k limits are enforced."""
    retriever = HybridRetriever.from_chunks(sample_chunks, embedder=dummy_embedder)
    
    # Test final k limit
    results = retriever.search("motor calibration", k=1)
    assert len(results) <= 1
    
    # Test with larger k than available documents
    results = retriever.search("motor", k=10)
    assert len(results) <= len(sample_chunks)


def test_snippet_highlights_query_terms(sample_chunks, dummy_embedder):
    """Test that snippet includes query terms and is within length limit."""
    retriever = HybridRetriever.from_chunks(sample_chunks, embedder=dummy_embedder)
    
    results = retriever.search("torque M3", k=5)
    
    # Find result containing "torque"
    torque_result = None
    for result in results:
        if "torque" in result['snippet'].lower():
            torque_result = result
            break
    
    assert torque_result is not None
    assert len(torque_result['snippet']) <= 240
    assert "torque" in torque_result['snippet'].lower() or "m3" in torque_result['snippet'].lower()


def test_tokenize_deterministic_unicode_safe():
    """Test that tokenization is deterministic and handles unicode."""
    text1 = "Hello, wörld! This is tëst 123."
    text2 = "Hello, wörld! This is tëst 123."
    
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    # Should be deterministic
    assert tokens1 == tokens2
    
    # Should handle unicode
    unicode_text = "café naïve résumé"
    tokens = tokenize(unicode_text)
    assert len(tokens) > 0
    assert all(isinstance(token, str) for token in tokens)
    
    # Should be lowercase
    assert all(token.islower() for token in tokens if token.isalpha())


def test_dummy_embedder_is_deterministic(dummy_embedder):
    """Test that DummyEmbedder produces identical vectors for repeated calls."""
    texts = ["Hello world", "Machine learning"]
    
    embeddings1 = dummy_embedder.encode(texts)
    embeddings2 = dummy_embedder.encode(texts)
    
    # Should be deterministic within tolerance
    assert np.allclose(embeddings1, embeddings2, atol=1e-8)
    
    # Should have correct shape
    assert embeddings1.shape == (2, dummy_embedder.dimension)


def test_empty_or_whitespace_query(retriever):
    """Test that empty or whitespace queries return empty list quickly."""
    # Empty query
    start_time = time.time()
    results = retriever.search("")
    end_time = time.time()
    
    assert results == []
    assert (end_time - start_time) < 0.1  # Should be very fast
    
    # Whitespace only
    results = retriever.search("   \n\t  ")
    assert results == []
    
    # Query with no valid tokens
    results = retriever.search("!@#$%")
    assert results == []


def test_rerank_ordering_stable_on_ties(sample_chunks, dummy_embedder):
    """Test that tie breaks are resolved by chunk_id ascending."""
    # Create a scenario where we can force RRF score ties
    # by manually patching the rrf function to return identical scores
    from src.retrieve import rrf as original_rrf
    import src.retrieve
    
    def mock_rrf(ranks, k=60):
        # Return identical scores to force a tie
        return {chunk_id: 0.5 for chunk_id in ranks.keys()}
    
    # Temporarily replace the rrf function
    src.retrieve.rrf = mock_rrf
    
    try:
        chunks_with_ties = [
            {"chunk_id": "z_last", "doc_id": "d1", "path": "p1", "title": "Test", "text": "different content A"},
            {"chunk_id": "a_first", "doc_id": "d2", "path": "p2", "title": "Test", "text": "different content B"},
            {"chunk_id": "m_middle", "doc_id": "d3", "path": "p3", "title": "Test", "text": "different content C"}
        ]
        
        retriever = HybridRetriever.from_chunks(chunks_with_ties, embedder=dummy_embedder)
        results = retriever.search("content", k=10)
        
        # Should be sorted by chunk_id ascending when RRF scores are tied
        chunk_ids = [r['chunk_id'] for r in results]
        sorted_chunk_ids = sorted(chunk_ids)
        assert chunk_ids == sorted_chunk_ids
        
    finally:
        # Restore original rrf function
        src.retrieve.rrf = original_rrf


@pytest.mark.slow
def test_latency_smoke(dummy_embedder):
    """Test search latency with ~200 chunks (marked slow, can be skipped)."""
    if os.getenv('SKIP_SLOW_TESTS'):
        pytest.skip("Slow test skipped via environment variable")
    
    # Create larger corpus
    chunks = []
    for i in range(200):
        chunks.append({
            "chunk_id": f"doc_{i}:0",
            "doc_id": f"doc_{i}",
            "path": f"path_{i}",
            "title": f"Title {i}",
            "text": f"This is document number {i} with various content about topic {i % 10}"
        })
    
    retriever = HybridRetriever.from_chunks(chunks, embedder=dummy_embedder)
    
    # Time the search
    start_time = time.time()
    results = retriever.search("document topic", k=20)
    end_time = time.time()
    
    search_time = end_time - start_time
    assert search_time < 0.5  # Should complete under 0.5 seconds
    assert len(results) > 0


def test_metadata_passthrough(sample_chunks, dummy_embedder):
    """Test that results carry metadata from input chunks."""
    retriever = HybridRetriever.from_chunks(sample_chunks, embedder=dummy_embedder)
    results = retriever.search("motor", k=5)
    
    # Find the motor result
    motor_result = None
    for result in results:
        if result['chunk_id'] == 'd1:0':
            motor_result = result
            break
    
    assert motor_result is not None
    assert motor_result['doc_id'] == 'd1'
    assert motor_result['path'] == 'p1'
    assert motor_result['title'] == 'Motor M3'
    assert motor_result['chunk_id'] == 'd1:0'


def test_make_snippet_centering():
    """Test snippet creation with query centering."""
    long_text = "This is a very long text that contains many words. " * 10 + " KEYWORD " + "More text follows here. " * 10
    
    # Should center around keyword
    snippet = make_snippet(long_text, "KEYWORD", max_len=50)
    assert "KEYWORD" in snippet
    assert len(snippet) <= 50 + 3  # +3 for potential ellipsis
    
    # Should fallback to head if no match
    snippet_no_match = make_snippet(long_text, "NOMATCH", max_len=50)
    assert snippet_no_match.startswith("This is a very long")
    

def test_rrf_calculation():
    """Test RRF score calculation with known values."""
    ranks = {
        "item1": [0],  # rank 1 only in first ranker
        "item2": [1, 2]  # rank 2 in first, rank 3 in second
    }
    
    scores = rrf(ranks, k=60)
    
    # item1: 1/(60+0+1) = 1/61 ≈ 0.0164
    # item2: 1/(60+1+1) + 1/(60+2+1) = 1/62 + 1/63 ≈ 0.0161 + 0.0159 = 0.0320
    
    assert scores["item2"] > scores["item1"]  # item2 should score higher
    assert abs(scores["item1"] - 1/61) < 1e-6
    assert abs(scores["item2"] - (1/62 + 1/63)) < 1e-6


def test_close_method(retriever):
    """Test that close method can be called without error."""
    retriever.close()  # Should not raise any exception