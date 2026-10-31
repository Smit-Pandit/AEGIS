# AEGIS // Project Chronos: Multimodal Incident Agentic RAG
### 100% Local MongoDB Search Engine & Epistemic Guardrails (NEXUS Hackathon)

AEGIS is an enterprise-grade SRE incident investigation console powered by a **100% Local MongoDB Search Engine** running natively on `localhost:27017` without requiring Atlas Cloud or Docker.

---

## Architecture Overview

```plaintext
nexus-hackathon/
├── backend/
│   ├── .env                  # API keys & local MongoDB URI
│   ├── db.py                 # Local MongoDB connection, indexes, vector + explain pipelines
│   ├── seed_and_ingest.py    # Generates 6 multimodal files + embeds into MongoDB
│   ├── retrieval.py          # Vector, BM25 Keyword, Hybrid RRF, & Metadata filtering
│   ├── agent.py              # Groq query router, conflict detector, epistemic guardrail
│   ├── main.py               # FastAPI server (Port 8000)
│   └── demo_dataset/         # Auto-generated Markdown, JSON, and PNG files
└── frontend/                 # React + Tailwind 3-Pane Developer Console (Port 3000)
```

---

## Key Features

1. **100% Local MongoDB Search Engine (`backend/db.py`)**:
   - **Metadata Pre-Filtering**: Compound B-Tree index on `metadata.service`, `metadata.source_type`, and `metadata.owner`.
   - **BM25 Keyword Search**: Native MongoDB `$text` index with weighted terms (10x for exact error codes like `ERR_REDIS_POOL_8821` and `PR #412`).
   - **Native Vector Cosine Aggregation**: MongoDB Aggregation pipeline using `$reduce` dot-product cosine similarity over 768-dim normalized embedding vectors.
   - **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF}(d) = \frac{1}{60 + r_{\text{vector}}(d)} + \frac{1}{60 + r_{\text{keyword}}(d)}$$
   - **Explainability**: Direct integration with MongoDB `db.command("explain", ..., verbosity="executionStats")`.

2. **Curated Multimodal Dataset (`backend/seed_and_ingest.py`)**:
   - `arch_spec_v1.md`: Mandates `redis_max_connections: 500` and `timeout_sla: 30s` (owner: `team-core`).
   - `runbook_policy_v2.md`: Mandates `timeout_sla: 5s` under PCI-DSS 4.0 (owner: `team-security`). **[Planted Conflict]**
   - `incident_2026_10_18.json`: SEV-1 outage log with error `ERR_REDIS_POOL_8821` when connections hit 50/50 (owner: `team-sre`).
   - `git_pr_412_diff.md`: Commit `a8f92b` by `@dev-alex` reducing pool from 500 to 50 (owner: `team-core`).
   - `kafka_downstream_log.json`: Secondary incident log showing `order-service` failing with `ERR_KAFKA_OFFSET_902` (owner: `team-orders`).
   - `grafana_memory_spike.png`: Matplotlib dark-theme telemetry chart showing `payment-node-03` hitting 99% pool saturation at 14:21 UTC.

3. **Agentic RAG Workflow & Epistemic Guardrails (`backend/agent.py`)**:
   - **Query Decomposition**: Splits multi-hop prompts into targeted sub-queries.
   - **Parallel Hybrid Retrieval**: Queries MongoDB vector & text pipelines across all sub-queries.
   - **Epistemic Guardrail**: Refuses generation (`status: "ABSTAINED"`) when maximum vector score is below 0.55 or query falls outside corpus domain (0% hallucination).
   - **Cross-Source Conflict Detection**: Identifies contradictory policies (e.g. 30s vs 5s SLA) and cites passages side-by-side.
   - **Grounded Artefact Generation**: Synthesizes an Engineering Risk & Root-Cause Brief where every statement carries a clickable inline citation tag (`[CIT:chk_XX]`).

4. **3-Pane Developer Console UI (`frontend/`)**:
   - **Left Pane**: Live file uploader (`POST /api/upload`), metadata pre-filters, and corpus explorer.
   - **Center Pane**: 3 one-click demo presets, custom query bar, status banners, conflict diffs, epistemic refusal cards, and cited brief.
   - **Right Pane**: Citation inspector with exact character offset highlighting, visual telemetry chart preview, score metrics, and genuine MongoDB `.explain("executionStats")` viewer.
   - **Retrieval Evaluation Lab**: Live benchmark suite demonstrating Hybrid RRF achieving **100% Recall@3**.

---

## Running the Project

### Prerequisites
- Local MongoDB running on `mongodb://localhost:27017/`
- Python 3.11+ / Virtualenv `hackathon`
- Node.js v18+

### 1. Backend Server (Port 8000)
```powershell
.\hackathon\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend Console (Port 3000)
```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 3000
```
Visit: **`http://localhost:3000`**

### 3. Re-seed Demo Dataset (Anytime)
```powershell
.\hackathon\Scripts\python.exe .\backend\seed_and_ingest.py
```
Or click the **"Reset"** button directly in the UI header!

