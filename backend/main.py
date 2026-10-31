import os
import sys
import json
import base64
import re
import uuid
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# Ensure parent and backend dirs are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.db import (
    raw_sources_col,
    chunks_col,
    eval_col,
    client,
    DB_NAME,
    DEFAULT_CORPUS_ID,
    init_local_db,
)
from backend.seed_and_ingest import reindex_all_existing_sources, chunk_text_with_offsets, extract_identifiers, process_image_asset
from backend.retrieval import evaluate_retrieval, compute_content_hash, get_embedding, INDEX_VERSION
from backend.agent import run_agentic_investigation

app = FastAPI(
    title="AEGIS Enterprise Multimodal RAG API",
    description="Agentic Multimodal RAG with Strict Corpus Isolation & Local MongoDB Search Engine",
    version="2.0.0",
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize indexes on startup
@app.on_event("startup")
def startup_event():
    init_local_db()


class InvestigateRequest(BaseModel):
    query: str
    corpus_id: Optional[str] = DEFAULT_CORPUS_ID
    metadata_filter: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


@app.get("/api/health")
def health_check(corpus_id: str = Query(DEFAULT_CORPUS_ID)):
    try:
        client.admin.command("ping")
        mongo_ok = True
    except Exception:
        mongo_ok = False

    raw_count = raw_sources_col.count_documents({"corpus_id": corpus_id})
    chunk_count = chunks_col.count_documents({"corpus_id": corpus_id})
    total_chunks = chunks_col.count_documents({})
    eval_count = eval_col.count_documents({})

    return {
        "status": "online",
        "mongodb_connected": mongo_ok,
        "database": DB_NAME,
        "active_corpus_id": corpus_id,
        "raw_sources_count": raw_count,
        "knowledge_chunks_count": chunk_count,
        "total_chunks_across_corpora": total_chunks,
        "eval_suite_count": eval_count,
        "index_version": INDEX_VERSION,
    }


@app.post("/api/investigate")
def investigate(req: InvestigateRequest, x_corpus_id: Optional[str] = Header(None)):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string is required")

    effective_corpus = x_corpus_id or req.corpus_id or DEFAULT_CORPUS_ID
    request_id = req.request_id or str(uuid.uuid4())

    try:
        response = run_agentic_investigation(
            user_query=req.query.strip(),
            corpus_id=effective_corpus,
            client_filter=req.metadata_filter,
            request_id=request_id,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")


@app.get("/api/evaluate")
def evaluate(corpus_id: str = Query(DEFAULT_CORPUS_ID), top_k: int = Query(3)):
    try:
        results = evaluate_retrieval(corpus_id=corpus_id, top_k=top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/api/sources")
def get_sources(corpus_id: str = Query(DEFAULT_CORPUS_ID)):
    sources = list(raw_sources_col.find({"corpus_id": corpus_id}, {"_id": 0}))
    for s in sources:
        sid = s.get("source_id")
        s["chunk_count"] = chunks_col.count_documents({"corpus_id": corpus_id, "source_id": sid})
    return {"corpus_id": corpus_id, "sources": sources}


@app.get("/api/sources/{source_id}")
def get_source_details(source_id: str, corpus_id: str = Query(DEFAULT_CORPUS_ID)):
    source = raw_sources_col.find_one({"corpus_id": corpus_id, "source_id": source_id}, {"_id": 0})
    if not source:
        # Fallback without corpus_id filter if not found
        source = raw_sources_col.find_one({"source_id": source_id}, {"_id": 0})
    if not source:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found in corpus '{corpus_id}'")

    chunks = list(chunks_col.find({"source_id": source_id}, {"_id": 0, "embedding": 0}))
    return {
        "source": source,
        "chunks": chunks,
    }


@app.get("/api/image/{source_id}")
def get_image_file(source_id: str, corpus_id: str = Query(DEFAULT_CORPUS_ID)):
    source = raw_sources_col.find_one({"corpus_id": corpus_id, "source_id": source_id})
    if not source or not source.get("image_base64"):
        source = raw_sources_col.find_one({"source_id": source_id})
    if not source or not source.get("image_base64"):
        raise HTTPException(status_code=404, detail=f"Image '{source_id}' not found")

    img_bytes = base64.b64decode(source["image_base64"])
    mime = source.get("metadata", {}).get("mime_type", "image/png")
    return Response(content=img_bytes, media_type=mime)


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    corpus_id: str = Form(DEFAULT_CORPUS_ID),
    service: str = Form("general"),
    owner: str = Form("user"),
    source_type: str = Form("custom_upload"),
):
    """Live file upload with strict corpus isolation, exact character offsets, and multimodal chunking."""
    try:
        content_bytes = await file.read()
        filename = file.filename
        ext = filename.split(".")[-1].lower() if "." in filename else "txt"
        clean_fn = re.sub(r"[^a-zA-Z0-9]", "_", filename)[:10]

        # 1. Multimodal Images
        if ext in ["png", "jpg", "jpeg", "webp", "gif"]:
            processed = process_image_asset(filename, content_bytes, corpus_id=corpus_id, service=service, owner=owner, ext=ext)
            raw_sources_col.update_one(
                {"corpus_id": corpus_id, "source_id": filename},
                {"$set": processed["raw_doc"]},
                upsert=True,
            )
            chunks_col.delete_many({"corpus_id": corpus_id, "source_id": filename})
            chunks_col.insert_one(processed["chunk_doc"])

        # 2. Textual Documents
        else:
            text_content = content_bytes.decode("utf-8", errors="replace")
            content_hash = compute_content_hash(text_content)
            raw_doc = {
                "corpus_id": corpus_id,
                "source_id": filename,
                "file_name": filename,
                "file_type": ext,
                "title": filename,
                "content": text_content,
                "content_hash": content_hash,
                "metadata": {
                    "service": service,
                    "owner": owner,
                    "source_type": source_type,
                    "mime_type": f"text/{ext}",
                },
            }
            raw_sources_col.update_one(
                {"corpus_id": corpus_id, "source_id": filename},
                {"$set": raw_doc},
                upsert=True,
            )

            chunks_col.delete_many({"corpus_id": corpus_id, "source_id": filename})
            parsed_chunks = chunk_text_with_offsets(text_content)

            for idx, c in enumerate(parsed_chunks, start=1):
                cid = f"chk_{clean_fn}_{idx:02d}"
                codes = extract_identifiers(c["content"])
                emb = get_embedding(c["content"])
                c_hash = compute_content_hash(c["content"])

                chunk_doc = {
                    "corpus_id": corpus_id,
                    "chunk_id": cid,
                    "source_id": filename,
                    "source_title": f"{filename}: {c.get('section', f'Section {idx}')}",
                    "content": c["content"],
                    "content_hash": c_hash,
                    "char_start": c["char_start"],
                    "char_end": c["char_end"],
                    "embedding": emb,
                    "embedding_version": INDEX_VERSION,
                    "metadata": {
                        "service": service,
                        "owner": owner,
                        "source_type": source_type,
                        "extracted_codes": codes,
                        "has_image": False,
                    },
                }
                chunks_col.insert_one(chunk_doc)

        return {
            "status": "success",
            "corpus_id": corpus_id,
            "source_id": filename,
            "message": f"File '{filename}' successfully indexed into corpus '{corpus_id}'.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/api/clear-all")
def clear_all(corpus_id: Optional[str] = None):
    """Wipes documents and chunks from MongoDB. Scoped to corpus_id if provided."""
    query = {"corpus_id": corpus_id} if corpus_id else {}
    raw_sources_col.delete_many(query)
    chunks_col.delete_many(query)
    if not corpus_id:
        eval_col.delete_many({})
    return {
        "status": "success",
        "cleared_corpus": corpus_id or "all",
        "message": f"Cleared data from {'corpus ' + corpus_id if corpus_id else 'all corpora'}.",
    }


@app.post("/api/reset-seed")
def reset_seed(corpus_id: str = Query(DEFAULT_CORPUS_ID)):
    """Reindexes existing sources in MongoDB."""
    try:
        reindex_all_existing_sources(target_corpus_id=corpus_id)
        return {"status": "success", "corpus_id": corpus_id, "message": "Corpus sources reindexed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


# Serve static frontend if built in frontend/dist
dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = os.path.join(dist_dir, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
