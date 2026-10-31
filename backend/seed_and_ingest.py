import os
import sys
import json
import base64
import re
import time
import hashlib
from typing import Dict, Any, List, Optional

# Ensure parent and backend dirs are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.db import (
    init_local_db,
    raw_sources_col,
    chunks_col,
    DEFAULT_CORPUS_ID,
)
from backend.retrieval import get_embedding, compute_content_hash, INDEX_VERSION
from backend.llm_client import call_gemini_multimodal

DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_dataset"))


def extract_identifiers(text: str) -> List[str]:
    """Exhaustive extraction of exact identifiers, error codes, PR tags, entity names, and symbols."""
    # 1. Alphanumeric uppercase codes with underscores, hyphens, or digits
    codes = re.findall(r"[A-Za-z0-9_]{3,40}", text)
    filtered = [
        c for c in set(codes)
        if any(char.isdigit() for char in c) or "_" in c or "-" in c
    ]

    # 2. Backticked tokens
    backticked = re.findall(r"`([^`]+)`", text)

    # 3. Capitalized Proper Names (e.g. Charizard, Pikachu, Gengar, Mewtwo)
    proper_names = re.findall(r"\b[A-Z][a-z]{3,20}\b", text)

    # Combine unique tokens
    all_tokens = list(dict.fromkeys(filtered + backticked + proper_names))
    return all_tokens[:30]


def chunk_text_with_offsets(content: str, max_chunk_chars: int = 650) -> List[Dict[str, Any]]:
    """Chunks text while computing mathematically exact character offsets (char_start, char_end).
    Guarantees: content[char_start:char_end] == chunk['content']
    Handles JSON arrays, dictionaries, markdown headings, and text blocks.
    """
    chunks = []
    content_len = len(content)
    stripped = content.strip()

    # 1. Structured JSON (Array of objects or Dict containing list of objects)
    if (stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]")):
        try:
            parsed = json.loads(content)
            items = []
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        items = v
                        break

            if items:
                curr_search_pos = 0
                for idx, item in enumerate(items, start=1):
                    # Find exact substring boundary in content for each item
                    # First try finding key identifier from the item
                    item_keys = [str(item[k]) for k in ["species_id", "id", "event_code", "name", "rule_id"] if k in item]
                    c_start = -1
                    if item_keys:
                        for key_val in item_keys:
                            key_pos = content.find(f'"{key_val}"', curr_search_pos)
                            if key_pos != -1:
                                # Rewind to the opening brace '{' before key_pos
                                brace_pos = content.rfind("{", curr_search_pos, key_pos)
                                if brace_pos != -1:
                                    c_start = brace_pos
                                    break

                    # Fallback to serialized snippet search
                    if c_start == -1:
                        c_start = content.find("{", curr_search_pos)

                    if c_start != -1:
                        # Find matching closing brace
                        depth = 0
                        in_str = False
                        esc = False
                        c_end = -1
                        for p in range(c_start, content_len):
                            char = content[p]
                            if char == '"' and not esc:
                                in_str = not in_str
                            elif char == "\\" and in_str:
                                esc = not esc
                                continue
                            elif not in_str:
                                if char == "{":
                                    depth += 1
                                elif char == "}":
                                    depth -= 1
                                    if depth == 0:
                                        c_end = p + 1
                                        break
                            esc = False

                        if c_end != -1:
                            item_content = content[c_start:c_end]
                            name = item.get("name") or item.get("species_id") or item.get("event_code") or f"Record {idx}"
                            chunks.append({
                                "content": item_content,
                                "char_start": c_start,
                                "char_end": c_end,
                                "section": f"Record {idx}: {name}",
                            })
                            curr_search_pos = c_end

                if chunks:
                    return chunks
        except Exception as e:
            print(f"[JSON Chunking Error] {e}")

    # 2. Markdown or Text Paragraph / Section Chunking
    pattern = re.compile(r"(\n#{1,4}\s+[^\n]+\n|\n\n+)")
    parts = pattern.split(content)
    curr_pos = 0
    buffer = ""
    buf_start = 0

    for part in parts:
        if not part:
            continue
        part_len = len(part)

        if len(buffer) + part_len > max_chunk_chars and buffer.strip():
            clean_chunk = buffer.strip()
            real_start = content.find(clean_chunk, buf_start)
            if real_start == -1:
                real_start = buf_start
            real_end = real_start + len(clean_chunk)
            first_line = clean_chunk.split("\n")[0].replace("#", "").strip()[:40] or f"Section {len(chunks) + 1}"
            chunks.append({
                "content": clean_chunk,
                "char_start": real_start,
                "char_end": real_end,
                "section": first_line,
            })
            buffer = part
            buf_start = curr_pos
        else:
            if not buffer:
                buf_start = curr_pos
            buffer += part

        curr_pos += part_len

    if buffer.strip():
        clean_chunk = buffer.strip()
        real_start = content.find(clean_chunk, buf_start)
        if real_start == -1:
            real_start = buf_start
        real_end = real_start + len(clean_chunk)
        first_line = clean_chunk.split("\n")[0].replace("#", "").strip()[:40] or f"Section {len(chunks) + 1}"
        chunks.append({
            "content": clean_chunk,
            "char_start": real_start,
            "char_end": real_end,
            "section": first_line,
        })

    if not chunks and content:
        chunks.append({
            "content": content,
            "char_start": 0,
            "char_end": len(content),
            "section": "Complete Document",
        })

    return chunks


def process_image_asset(
    source_id: str,
    image_bytes: bytes,
    corpus_id: str = DEFAULT_CORPUS_ID,
    service: str = "general",
    owner: str = "user",
    ext: str = "png",
) -> Dict[str, Any]:
    """Uses multimodal LLM (Gemini) to generate rich semantic captions and entity tags for images."""
    b64_payload = base64.b64encode(image_bytes).decode("utf-8")
    content_hash = compute_content_hash(b64_payload)

    # 1. Ask Gemini to provide a rich description, entity identification, and key attributes
    prompt = (
        "Identify the primary subject, character, or entity depicted in this image.\n"
        "Return a concise 2-sentence description including: entity name, visual attributes (color, pose, features), and category."
    )
    try:
        visual_desc = call_gemini_multimodal(b64_payload, mime_type=f"image/{ext}", prompt=prompt)
    except Exception as e:
        print(f"[Multimodal Gemini Analysis Error for {source_id}] {e}")
        visual_desc = f"Visual image asset: {source_id}"

    # 2. Extract entity names from visual description
    extracted_entities = extract_identifiers(visual_desc)
    if source_id not in extracted_entities:
        extracted_entities.append(source_id)

    full_content = (
        f"# Visual Document: {source_id}\n"
        f"**Identified Subject & Telemetry:** {visual_desc}\n"
        f"- Format: {ext.upper()} | Size: {len(image_bytes)} bytes | Service: {service} | Owner: {owner}"
    )

    return {
        "raw_doc": {
            "corpus_id": corpus_id,
            "source_id": source_id,
            "file_name": source_id,
            "file_type": ext,
            "title": f"Image Asset: {source_id}",
            "content": full_content,
            "content_hash": content_hash,
            "image_base64": b64_payload,
            "metadata": {
                "service": service,
                "owner": owner,
                "source_type": "visual_telemetry",
                "mime_type": f"image/{ext}",
                "extracted_entities": extracted_entities,
                "visual_description": visual_desc,
            },
            "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "chunk_doc": {
            "corpus_id": corpus_id,
            "chunk_id": f"chk_{re.sub(r'[^a-zA-Z0-9]', '', source_id)[:8].lower()}_01",
            "source_id": source_id,
            "source_title": f"Visual Asset: {source_id}",
            "content": full_content,
            "content_hash": compute_content_hash(full_content),
            "char_start": 0,
            "char_end": len(full_content),
            "embedding": get_embedding(full_content),
            "embedding_version": INDEX_VERSION,
            "metadata": {
                "service": service,
                "owner": owner,
                "source_type": "visual_telemetry",
                "extracted_codes": extracted_entities,
                "has_image": True,
                "visual_description": visual_desc,
            },
            "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def ingest_file(
    filepath: str,
    corpus_id: str = DEFAULT_CORPUS_ID,
    service: str = "general",
    owner: str = "user",
    source_type: str = "document",
) -> int:
    """Ingests any user file into MongoDB with strict corpus isolation, exact offsets, and multimodal support."""
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filename)[1].lower().replace(".", "")
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    file_size = os.path.getsize(filepath)

    # 1. Multimodal Images (PNG, JPG, WEBP, JPEG)
    if ext in ["png", "jpg", "jpeg", "webp", "gif"]:
        with open(filepath, "rb") as f:
            raw_bytes = f.read()

        processed = process_image_asset(filename, raw_bytes, corpus_id=corpus_id, service=service, owner=owner, ext=ext)
        raw_sources_col.update_one(
            {"corpus_id": corpus_id, "source_id": filename},
            {"$set": processed["raw_doc"]},
            upsert=True,
        )
        chunks_col.delete_many({"corpus_id": corpus_id, "source_id": filename})
        chunks_col.insert_one(processed["chunk_doc"])
        return 1

    # 2. Textual Documents (Markdown, JSON, Text, Logs)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    content_hash = compute_content_hash(content)

    raw_sources_col.update_one(
        {"corpus_id": corpus_id, "source_id": filename},
        {"$set": {
            "corpus_id": corpus_id,
            "source_id": filename,
            "filename": filename,
            "file_type": ext,
            "content": content,
            "content_hash": content_hash,
            "metadata": {
                "service": service,
                "owner": owner,
                "source_type": source_type,
                "file_size": file_size,
                "mime_type": f"text/{ext}",
            },
            "uploaded_at": now_iso,
        }},
        upsert=True,
    )

    chunks_col.delete_many({"corpus_id": corpus_id, "source_id": filename})
    parsed_chunks = chunk_text_with_offsets(content)
    inserted = 0

    clean_prefix = re.sub(r"[^a-zA-Z0-9]", "", filename)[:8].lower()
    for idx, c in enumerate(parsed_chunks, start=1):
        cid = f"chk_{clean_prefix}_{idx:02d}"
        codes = extract_identifiers(c["content"])
        embedding = get_embedding(c["content"])
        c_hash = compute_content_hash(c["content"])

        chunks_col.insert_one({
            "corpus_id": corpus_id,
            "chunk_id": cid,
            "source_id": filename,
            "source_title": f"{filename}: {c.get('section', f'Section {idx}')}",
            "content": c["content"],
            "content_hash": c_hash,
            "char_start": c["char_start"],
            "char_end": c["char_end"],
            "embedding": embedding,
            "embedding_version": INDEX_VERSION,
            "metadata": {
                "service": service,
                "owner": owner,
                "source_type": source_type,
                "extracted_codes": codes,
                "has_image": False,
            },
            "ingested_at": now_iso,
        })
        inserted += 1

    return inserted


def reindex_all_existing_sources(target_corpus_id: Optional[str] = None):
    """Re-chunks and re-embeds all existing raw sources in MongoDB with full vision comprehension and offsets."""
    query = {"corpus_id": target_corpus_id} if target_corpus_id else {}
    sources = list(raw_sources_col.find(query))
    print(f"[Reindex Engine] Starting reindexing for {len(sources)} documents...")

    for doc in sources:
        sid = doc.get("source_id")
        cid = doc.get("corpus_id", DEFAULT_CORPUS_ID)
        ftype = doc.get("file_type", "")
        meta = doc.get("metadata", {})
        service = meta.get("service", "general")
        owner = meta.get("owner", "user")

        if doc.get("image_base64"):
            # Image asset
            raw_bytes = base64.b64decode(doc["image_base64"])
            processed = process_image_asset(sid, raw_bytes, corpus_id=cid, service=service, owner=owner, ext=ftype or "png")
            raw_sources_col.update_one({"corpus_id": cid, "source_id": sid}, {"$set": processed["raw_doc"]})
            chunks_col.delete_many({"corpus_id": cid, "source_id": sid})
            chunks_col.insert_one(processed["chunk_doc"])
            print(f"  [Reindexed Image] {sid} with vision description.")
        else:
            # Textual document
            content = doc.get("content", "")
            if not content:
                continue
            chunks_col.delete_many({"corpus_id": cid, "source_id": sid})
            parsed_chunks = chunk_text_with_offsets(content)
            clean_prefix = re.sub(r"[^a-zA-Z0-9]", "", sid)[:8].lower()
            for idx, c in enumerate(parsed_chunks, start=1):
                chunk_id = f"chk_{clean_prefix}_{idx:02d}"
                codes = extract_identifiers(c["content"])
                chunks_col.insert_one({
                    "corpus_id": cid,
                    "chunk_id": chunk_id,
                    "source_id": sid,
                    "source_title": f"{sid}: {c.get('section', f'Section {idx}')}",
                    "content": c["content"],
                    "content_hash": compute_content_hash(c["content"]),
                    "char_start": c["char_start"],
                    "char_end": c["char_end"],
                    "embedding": get_embedding(c["content"]),
                    "embedding_version": INDEX_VERSION,
                    "metadata": {
                        "service": service,
                        "owner": owner,
                        "source_type": meta.get("source_type", "document"),
                        "extracted_codes": codes,
                        "has_image": False,
                    },
                    "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
            print(f"  [Reindexed Text] {sid} into {len(parsed_chunks)} chunks.")

    print("[Reindex Engine] Reindexing complete.")


if __name__ == "__main__":
    reindex_all_existing_sources()
