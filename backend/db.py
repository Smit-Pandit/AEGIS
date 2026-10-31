import os
import sys
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, TEXT

# Ensure parent and backend dirs are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "aegis_rag")
DEFAULT_CORPUS_ID = os.getenv("DEFAULT_CORPUS_ID", "default")
USE_ATLAS_VECTOR = os.getenv("USE_ATLAS_VECTOR_SEARCH", "false").lower() == "true"

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = client[DB_NAME]

# Collections
raw_sources_col = db["raw_sources"]
chunks_col = db["knowledge_chunks"]
eval_col = db["eval_suite"]
corpus_meta_col = db["corpus_metadata"]


def init_local_db():
    """Initializes collections and enterprise indexes with strict corpus isolation."""
    try:
        client.admin.command("ping")
        print(f"[DB] Connected to MongoDB at {MONGODB_URI}{DB_NAME} (Atlas Vector Mode: {USE_ATLAS_VECTOR})")
    except Exception as e:
        print(f"[DB Warning] Could not ping MongoDB at {MONGODB_URI}: {e}")

    # 1. Corpus Isolation + Metadata Compound Index
    try:
        chunks_col.create_index(
            [
                ("corpus_id", ASCENDING),
                ("metadata.service", ASCENDING),
                ("metadata.source_type", ASCENDING),
                ("metadata.owner", ASCENDING),
            ],
            name="corpus_metadata_filter_idx",
            background=True,
        )
    except Exception as e:
        print(f"[DB Index Note] corpus_metadata_filter_idx: {e}")

    # 2. Chunk uniqueness and rapid retrieval per corpus
    try:
        chunks_col.create_index(
            [
                ("corpus_id", ASCENDING),
                ("chunk_id", ASCENDING),
            ],
            name="corpus_chunk_unique_idx",
            unique=True,
            background=True,
        )
    except Exception as e:
        print(f"[DB Index Note] corpus_chunk_unique_idx: {e}")

    # 3. Source document lookup index per corpus
    try:
        raw_sources_col.create_index(
            [
                ("corpus_id", ASCENDING),
                ("source_id", ASCENDING),
            ],
            name="corpus_source_unique_idx",
            unique=True,
            background=True,
        )
    except Exception as e:
        print(f"[DB Index Note] corpus_source_unique_idx: {e}")

    # 4. Native Full-Text BM25 Index for Hybrid Search (exact error codes + keywords)
    try:
        chunks_col.create_index(
            [
                ("metadata.extracted_codes", TEXT),
                ("source_title", TEXT),
                ("content", TEXT),
            ],
            weights={
                "metadata.extracted_codes": 10,
                "source_title": 5,
                "content": 1,
            },
            name="hybrid_keyword_idx",
            background=True,
        )
    except Exception as e:
        print(f"[DB Index Note] hybrid_keyword_idx: {e}")

    print("[DB] Local MongoDB Enterprise Indexes Configured & Verified!")


def build_vector_pipeline(
    query_vector: List[float],
    corpus_id: str = DEFAULT_CORPUS_ID,
    metadata_filter: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Builds MongoDB vector search pipeline with strict corpus namespace pre-filtering.
    Supports local exact Cosine Similarity on L2-normalized 768-dim vectors via $reduce dot-product.
    """
    match_stage = {"corpus_id": corpus_id}
    if metadata_filter:
        match_stage.update(metadata_filter)

    if USE_ATLAS_VECTOR:
        vs_stage = {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": max(50, top_k * 10),
            "limit": top_k,
            "filter": match_stage,
        }
        return [
            {"$vectorSearch": vs_stage},
            {"$addFields": {"vector_score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
        ]

    # Local Community Edition: exact Cosine Similarity on L2-normalized vectors
    return [
        {"$match": match_stage},
        {
            "$addFields": {
                "vector_score": {
                    "$reduce": {
                        "input": {"$range": [0, {"$size": "$embedding"}]},
                        "initialValue": 0.0,
                        "in": {
                            "$add": [
                                "$$value",
                                {
                                    "$multiply": [
                                        {"$arrayElemAt": ["$embedding", "$$this"]},
                                        {"$arrayElemAt": [query_vector, "$$this"]},
                                    ]
                                },
                            ]
                        },
                    }
                }
            }
        },
        {"$sort": {"vector_score": -1}},
        {"$limit": top_k},
        {"$project": {"embedding": 0}},
    ]


def build_keyword_pipeline(
    query_text: str,
    corpus_id: str = DEFAULT_CORPUS_ID,
    metadata_filter: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Native MongoDB Text Search aggregation pipeline with score and corpus pre-filter."""
    match_criteria = {
        "corpus_id": corpus_id,
        "$text": {"$search": query_text},
    }
    if metadata_filter:
        match_criteria.update(metadata_filter)

    return [
        {"$match": match_criteria},
        {
            "$addFields": {
                "keyword_score": {"$meta": "textScore"}
            }
        },
        {"$sort": {"keyword_score": -1}},
        {"$limit": top_k},
        {"$project": {"embedding": 0}},
    ]


def explain_vector_query(
    query_vector: List[float],
    corpus_id: str = DEFAULT_CORPUS_ID,
    metadata_filter: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Returns genuine MongoDB executionStats for the Vector Search pipeline."""
    pipeline = build_vector_pipeline(query_vector, corpus_id, metadata_filter, top_k)
    try:
        return db.command(
            "explain",
            {"aggregate": "knowledge_chunks", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
    except Exception as e:
        return {"error": f"Explain command error: {str(e)}", "pipeline": pipeline}


def explain_keyword_query(
    query_text: str,
    corpus_id: str = DEFAULT_CORPUS_ID,
    metadata_filter: Optional[Dict[str, Any]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Returns genuine MongoDB executionStats for the Keyword Search pipeline."""
    pipeline = build_keyword_pipeline(query_text, corpus_id, metadata_filter, top_k)
    try:
        return db.command(
            "explain",
            {"aggregate": "knowledge_chunks", "pipeline": pipeline, "cursor": {}},
            verbosity="executionStats",
        )
    except Exception as e:
        return {"error": f"Explain command error: {str(e)}", "pipeline": pipeline}
