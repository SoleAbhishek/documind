import math
from typing import List, Dict, Any, Sequence


def is_chunk_relevant(chunk: Dict[str, Any], expected_keywords: List[str]) -> bool:
    """
    Determines if a retrieved chunk contains the ground truth evidence.
    Returns True if at least 50% of expected key phrases/terms appear in the chunk text.
    """
    if not expected_keywords:
        return True

    text = chunk.get("text", "").lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in text)
    # Require at least one match, or >= 50% of keywords if multiple keywords specified
    threshold = max(1, len(expected_keywords) // 2)
    return matches >= threshold


def compute_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_keywords: List[str],
    k: int = 3
) -> float:
    """
    Recall@K: 1.0 if at least one relevant chunk containing ground-truth evidence is found in top-K, else 0.0.
    """
    if not expected_keywords:
        return 1.0

    top_k = retrieved_chunks[:k]
    for chunk in top_k:
        if is_chunk_relevant(chunk, expected_keywords):
            return 1.0
    return 0.0


def compute_precision_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_keywords: List[str],
    k: int = 3
) -> float:
    """
    Precision@K: Fraction of top-K retrieved chunks that are relevant.
    Score in [0.0, 1.0].
    """
    if not expected_keywords:
        return 1.0
    if k <= 0:
        return 0.0

    top_k = retrieved_chunks[:k]
    if not top_k:
        return 0.0

    relevant_count = sum(1 for chunk in top_k if is_chunk_relevant(chunk, expected_keywords))
    return round(relevant_count / len(top_k), 3)


def compute_mrr(
    retrieved_chunks: List[Dict[str, Any]],
    expected_keywords: List[str]
) -> float:
    """
    Mean Reciprocal Rank (MRR): 1 / rank of the first relevant chunk found.
    Returns 0.0 if no relevant chunk is found.
    """
    if not expected_keywords:
        return 1.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if is_chunk_relevant(chunk, expected_keywords):
            return round(1.0 / rank, 3)
    return 0.0


def compute_ndcg_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    expected_keywords: List[str],
    k: int = 3
) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K):
    Measures ranking quality with logarithmic position discounting:
      DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
      IDCG@K = ideal DCG where all relevant documents are ranked at the top
    Score in [0.0, 1.0].
    """
    if not expected_keywords:
        return 1.0
    if k <= 0:
        return 0.0

    top_k = retrieved_chunks[:k]
    if not top_k:
        return 0.0

    dcg = 0.0
    for i, chunk in enumerate(top_k, start=1):
        rel = 1.0 if is_chunk_relevant(chunk, expected_keywords) else 0.0
        dcg += rel / math.log2(i + 1)

    # Count total relevant items available in retrieved set (up to k) for IDCG
    total_rel = sum(1 for chunk in top_k if is_chunk_relevant(chunk, expected_keywords))
    if total_rel == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, total_rel + 1))
    return round(dcg / idcg, 3) if idcg > 0 else 0.0
