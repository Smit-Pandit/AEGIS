import os
import sys
import re
import time
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv

# Ensure parent and backend dirs are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.db import (
    chunks_col,
    eval_col,
    DEFAULT_CORPUS_ID,
    build_vector_pipeline,
    build_keyword_pipeline,
    explain_vector_query,
    explain_keyword_query,
)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

INDEX_VERSION = "aegis_v2_hybrid"


def compute_content_hash(text: str) -> str:
    """Computes SHA-256 hash for document content to guarantee immutability and provenance."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dense_hash_embed(text: str, dim: int = 768) -> List[float]:
    """Generates a deterministic 768-dimensional normalized dense embedding for any text.
    Uses subword n-gram hashing and term frequency weights.
    Operates purely on language syntax and token representations.
    """
    cleaned = text.lower().strip()
    tokens = re.findall(r"[a-z0-9_]+", cleaned)
    if not tokens:
        tokens = [cleaned or "empty"]

    vec = np.zeros(dim, dtype=np.float32)

    for token in tokens:
        # Full token hash
        h_full = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
        idx_full = h_full % dim
        sign_full = 1.0 if ((h_full >> 8) & 1) else -1.0
        vec[idx_full] += sign_full * 1.5

        # Subword character n-grams (3-grams and 4-grams)
        if len(token) >= 3:
            for n in (3, 4):
                for i in range(len(token) - n + 1):
                    ngram = token[i : i + n]
                    h_ng = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                    idx_ng = h_ng % dim
                    sign_ng = 1.0 if ((h_ng >> 8) & 1) else -1.0
                    vec[idx_ng] += sign_ng * 0.5

    # L2 normalize vector to unit sphere for exact cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    else:
        vec[0] = 1.0

    return [round(float(x), 6) for x in vec]


def get_embedding(text: str) -> List[float]:
    """Retrieves 768-dimensional embedding vector for query or text chunk."""
    return _dense_hash_embed(text, dim=768)


def vector_search(
    query_vector: List[float],
    corpus_id: str = DEFAULT_CORPUS_ID,
    metadata_filter: Optional[Dict[str, Any]] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Native MongoDB aggregation pipeline for cosine similarity search with corpus isolation."""
    pipeline = build_vector_pipeline(query_vector, corpus_id=corpus_id, metadata_filter=metadata_filter, top_k=limit)
    cursor = chunks_col.aggregate(pipeline)
    results = []
    now_ts = time.time()
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["retrieval_method"] = "vector"
        doc["retrieval_timestamp"] = now_ts
        doc["index_version"] = INDEX_VERSION
        results.append(doc)
    return results


def keyword_search(
    query_text: str,
    corpus_id: str = DEFAULT_CORPUS_ID,
    metadata_filter: Optional[Dict[str, Any]] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Native MongoDB $text search with BM25-derived textScore with corpus isolation."""
    pipeline = build_keyword_pipeline(query_text, corpus_id=corpus_id, metadata_filter=metadata_filter, top_k=limit)
    cursor = chunks_col.aggregate(pipeline)
    results = []
    now_ts = time.time()
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["retrieval_method"] = "keyword"
        doc["retrieval_timestamp"] = now_ts
        doc["index_version"] = INDEX_VERSION
        results.append(doc)
    return results


def structured_code_search(
    query_text: str,
    corpus_id: str = DEFAULT_CORPUS_ID,
    metadata_filter: Optional[Dict[str, Any]] = None,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """PRIORITY 2: Structured Retrieval & Exhaustive Identifier Lookup.
    Extracts identifiers, error codes, and named entities from query and performs exact index lookup.
    """
    # Extract identifiers, backticked codes, or capitalized proper nouns
    codes = re.findall(r"[A-Za-z0-9_]{3,40}", query_text)
    filtered = [c for c in set(codes) if any(ch.isdigit() for ch in c) or "_" in c or "-" in c]
    backticked = re.findall(r"`([^`]+)`", query_text)
    proper_names = re.findall(r"\b[A-Z][a-z]{3,20}\b", query_text)

    search_tokens = list(dict.fromkeys(filtered + backticked + proper_names))
    if not search_tokens:
        return []

    # Build Mongo filter enforcing corpus isolation
    query_filter: Dict[str, Any] = {"corpus_id": corpus_id}
    if metadata_filter:
        query_filter.update(metadata_filter)

    or_clauses = [
        {"metadata.extracted_codes": {"$in": search_tokens}},
    ]
    for token in search_tokens[:4]:
        or_clauses.append({"content": {"$regex": re.escape(token), "$options": "i"}})

    query_filter["$or"] = or_clauses

    now_ts = time.time()
    cursor = chunks_col.find(query_filter).limit(limit)
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        doc["retrieval_method"] = "structured"
        doc["retrieval_timestamp"] = now_ts
        doc["index_version"] = INDEX_VERSION
        results.append(doc)
    return results


def hybrid_rrf_search(
    query_text: str,
    corpus_id: str = DEFAULT_CORPUS_ID,
    metadata_filter: Optional[Dict[str, Any]] = None,
    top_k: int = 6,
    k_rrf: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
    structured_weight: float = 1.5,
) -> Dict[str, Any]:
    """Combines Vector search, BM25 Keyword search, and Structured Identifier search using calibrated RRF:
    RRF(d) = sum( weight_i / (k_rrf + rank_i) )
    Guarantees corpus-level namespace isolation, exact identifier matching, and immutable provenance.
    """
    t0 = time.time()
    query_vec = get_embedding(query_text)

    # 1. Vector Search
    v_results = vector_search(query_vec, corpus_id=corpus_id, metadata_filter=metadata_filter, limit=top_k * 2)

    # 2. Keyword Search
    clean_keyword = re.sub(r'["\']', " ", query_text).strip()
    k_results = keyword_search(clean_keyword, corpus_id=corpus_id, metadata_filter=metadata_filter, limit=top_k * 2)

    # 3. Structured Search (Exact identifier / entity lookup)
    s_results = structured_code_search(query_text, corpus_id=corpus_id, metadata_filter=metadata_filter, limit=top_k * 2)

    # 4. Multi-modal Intent Detection
    visual_intent = bool(re.search(r"\b(image|picture|photo|diagram|chart|visual|appearance|sprite|telemetry|look)\b", query_text, re.I))

    # 5. Reciprocal Rank Fusion
    rrf_map: Dict[str, Dict[str, Any]] = {}
    now_ts = time.time()

    # Vector
    for rank, doc in enumerate(v_results, start=1):
        cid = doc["chunk_id"]
        v_score = round(float(doc.get("vector_score", 0.0)), 4)
        if cid not in rrf_map:
            doc_copy = dict(doc)
            doc_copy["corpus_id"] = doc.get("corpus_id", corpus_id)
            doc_copy["vector_score"] = v_score
            doc_copy["keyword_score"] = 0.0
            doc_copy["structured_score"] = 0.0
            doc_copy["vector_rank"] = rank
            doc_copy["keyword_rank"] = None
            doc_copy["rrf_score"] = round(vector_weight / (k_rrf + rank), 5)
            doc_copy["retrieval_method"] = "hybrid_rrf"
            doc_copy["retrieval_timestamp"] = now_ts
            doc_copy["index_version"] = INDEX_VERSION
            doc_copy["content_hash"] = doc.get("content_hash", compute_content_hash(doc.get("content", "")))
            rrf_map[cid] = doc_copy
        else:
            rrf_map[cid]["vector_score"] = v_score
            rrf_map[cid]["vector_rank"] = rank
            rrf_map[cid]["rrf_score"] = round(rrf_map[cid]["rrf_score"] + (vector_weight / (k_rrf + rank)), 5)

    # Keyword
    for rank, doc in enumerate(k_results, start=1):
        cid = doc["chunk_id"]
        k_score = round(float(doc.get("keyword_score", 0.0)), 4)
        if cid not in rrf_map:
            doc_copy = dict(doc)
            doc_copy["corpus_id"] = doc.get("corpus_id", corpus_id)
            doc_copy["vector_score"] = 0.0
            doc_copy["keyword_score"] = k_score
            doc_copy["structured_score"] = 0.0
            doc_copy["vector_rank"] = None
            doc_copy["keyword_rank"] = rank
            doc_copy["rrf_score"] = round(keyword_weight / (k_rrf + rank), 5)
            doc_copy["retrieval_method"] = "hybrid_rrf"
            doc_copy["retrieval_timestamp"] = now_ts
            doc_copy["index_version"] = INDEX_VERSION
            doc_copy["content_hash"] = doc.get("content_hash", compute_content_hash(doc.get("content", "")))
            rrf_map[cid] = doc_copy
        else:
            rrf_map[cid]["keyword_score"] = k_score
            rrf_map[cid]["keyword_rank"] = rank
            rrf_map[cid]["rrf_score"] = round(rrf_map[cid]["rrf_score"] + (keyword_weight / (k_rrf + rank)), 5)

    # Structured
    for rank, doc in enumerate(s_results, start=1):
        cid = doc["chunk_id"]
        if cid not in rrf_map:
            doc_copy = dict(doc)
            doc_copy["corpus_id"] = doc.get("corpus_id", corpus_id)
            doc_copy["vector_score"] = 0.0
            doc_copy["keyword_score"] = 0.0
            doc_copy["structured_score"] = 1.0
            doc_copy["vector_rank"] = None
            doc_copy["keyword_rank"] = None
            doc_copy["rrf_score"] = round(structured_weight / (k_rrf + rank), 5)
            doc_copy["retrieval_method"] = "hybrid_rrf"
            doc_copy["retrieval_timestamp"] = now_ts
            doc_copy["index_version"] = INDEX_VERSION
            doc_copy["content_hash"] = doc.get("content_hash", compute_content_hash(doc.get("content", "")))
            rrf_map[cid] = doc_copy
        else:
            rrf_map[cid]["structured_score"] = 1.0
            rrf_map[cid]["rrf_score"] = round(rrf_map[cid]["rrf_score"] + (structured_weight / (k_rrf + rank)), 5)

    # Visual boost for multimodal queries
    if visual_intent:
        for cid, doc in rrf_map.items():
            if doc.get("metadata", {}).get("has_image"):
                doc["rrf_score"] = round(doc["rrf_score"] * 1.8, 5)

    ranked_docs = sorted(rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)
    selected = ranked_docs[:top_k]

    latency_ms = round((time.time() - t0) * 1000, 2)
    explain_data = explain_vector_query(query_vec, corpus_id=corpus_id, metadata_filter=metadata_filter, top_k=top_k)

    return {
        "results": selected,
        "diagnostics": {
            "corpus_id": corpus_id,
            "vector_candidates_count": len(v_results),
            "keyword_candidates_count": len(k_results),
            "structured_candidates_count": len(s_results),
            "selected_count": len(selected),
            "retrieval_latency_ms": latency_ms,
            "applied_filters": metadata_filter or {},
            "k_rrf": k_rrf,
            "index_version": INDEX_VERSION,
        },
        "explain": explain_data,
    }


def evaluate_retrieval(corpus_id: str = DEFAULT_CORPUS_ID, top_k: int = 3) -> Dict[str, Any]:
    """Enterprise Live Retrieval Evaluation Benchmark:
    Calculates Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and NDCG@K.
    """
    test_cases = list(eval_col.find({"corpus_id": corpus_id}))
    if not test_cases:
        test_cases = list(eval_col.find({}))
    if not test_cases:
        return {
            "total_queries": 0,
            "corpus_id": corpus_id,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "results": [],
        }

    total = len(test_cases)
    h_hits = 0
    v_hits = 0
    k_hits = 0
    reciprocal_ranks = []
    ndcg_scores = []
    query_eval_results = []

    for tc in test_cases:
        q = tc.get("query", "")
        expected_sids = set(tc.get("expected_source_ids", []))
        if not expected_sids:
            continue

        q_vec = get_embedding(q)
        v_res = vector_search(q_vec, corpus_id=corpus_id, limit=top_k)
        v_sources = {r.get("source_id") for r in v_res}
        if v_sources.intersection(expected_sids):
            v_hits += 1

        k_res = keyword_search(q, corpus_id=corpus_id, limit=top_k)
        k_sources = {r.get("source_id") for r in k_res}
        if k_sources.intersection(expected_sids):
            k_hits += 1

        h_search = hybrid_rrf_search(q, corpus_id=corpus_id, top_k=top_k)
        h_res = h_search.get("results", [])
        h_sources = [r.get("source_id") for r in h_res]

        rr = 0.0
        dcg = 0.0
        for rank, sid in enumerate(h_sources, start=1):
            if sid in expected_sids:
                if rr == 0.0:
                    rr = 1.0 / rank
                dcg += 1.0 / np.log2(rank + 1)

        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(expected_sids), top_k) + 1))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        if any(sid in expected_sids for sid in h_sources):
            h_hits += 1

        reciprocal_ranks.append(rr)
        ndcg_scores.append(ndcg)

        query_eval_results.append({
            "query": q,
            "expected_source_ids": list(expected_sids),
            "retrieved_source_ids": h_sources,
            "hit": any(sid in expected_sids for sid in h_sources),
            "rrf_score_top1": h_res[0].get("rrf_score", 0.0) if h_res else 0.0,
            "rr": round(rr, 4),
            "ndcg": round(ndcg, 4),
        })

    mrr = round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0
    mean_ndcg = round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0.0
    recall_at_k = round(h_hits / total, 4) if total > 0 else 0.0
    precision_at_k = round(h_hits / (total * top_k), 4) if total > 0 else 0.0

    return {
        "total_queries": total,
        "corpus_id": corpus_id,
        "top_k": top_k,
        "recall_at_k": recall_at_k,
        "hybrid_recall_at_k": recall_at_k,
        "vector_only_recall": round(v_hits / total, 4) if total > 0 else 0.0,
        "keyword_only_recall": round(k_hits / total, 4) if total > 0 else 0.0,
        "precision_at_k": precision_at_k,
        "mrr": mrr,
        "ndcg_at_k": mean_ndcg,
        "results": query_eval_results,
    }


if __name__ == "__main__":
    print("Testing hybrid retrieval...")
    res = hybrid_rrf_search("Charizard special move code", top_k=3)
    for r in res["results"]:
        print(r["chunk_id"], r["source_id"], "RRF:", r["rrf_score"])
