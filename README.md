# HKEx OMDD Agentic GraphRAG

**Adaptive Hybrid RAG + Kuzu GraphRAG for HKEX OMDD technical documentation**

An agentic Retrieval-Augmented Generation system that answers questions across the HKEX OMD Derivatives (OMDD) manual set using hybrid BM25 + dense vector retrieval, LLM reranking, grounded verification, and — for cross-document questions — Kuzu-based knowledge-graph expansion.

> **How this differs from the original Agentic RAG for HKEx OMDD:** the original system was a single-pass hybrid retriever (BM25 + Chroma dense search) with LLM reranking and grounding verification. This version adds a **Kuzu-backed evidence graph** built from LLM-extracted, provenance-tagged triples. For questions classified as `cross_document`, the graph is traversed to surface additional evidence chunks that keyword/vector search alone would miss, then merged with hybrid results before reranking. Graph triples are never used directly as answer evidence — they only locate supporting original-document chunks, preserving strict grounding.

## Key features

- **Hybrid retrieval** — BM25 (exact protocol terms) + dense vector search (semantic matching) via `EnsembleRetriever`
- **Adaptive GraphRAG** — Kuzu graph traversal is triggered only for `cross_document` questions, avoiding unnecessary latency on exact lookups
- **Evidence-backed graph** — every triple stores `source`, `page`, `chunk_id`, `evidence_text`, and `confidence`
- **Parameterized Cypher** — all runtime values bound via `$params`, immune to quote/newline injection in evidence text
- **Question classification** — routes `message_format`, `operational_procedure`, `connectivity`, and `cross_document` questions to the right source PDFs
- **Retry / decomposition loop** — failed retrieval rounds generate targeted sub-questions and retry up to `MAX_RETRIES`
- **Grounded verification** — a judge LLM checks answer support; unsupported answers abstain with "Do not know"
- **PII sanitization** — redacts IPs, emails, tokens, and credentials from LLM output
- **Offline evaluation harness** — `evaluation/evaluate.py` + `test_cases.json` benchmark pass rate, source recall, latency, and graph-route accuracy

## Architecture

```text
docs/*.pdf
   |
   v
indexing.py
   |- Chroma vector store (BGE embeddings)
   |- BM25 JSONL cache (page-aware chunks)
   |
   v
graph_indexing.py
   |- LLM extracts evidence-backed triples per chunk
   |- Kuzu graph with provenance (source/page/chunk_id/evidence/confidence)
   |
   v
retrieving.py (LangGraph)
   |- classify_question
   |- plan_queries
   |- retrieve_hybrid (BM25 + dense + per-source scoped)
   |- graph_seed_entities → graph_expand → fetch_graph_evidence   [cross_document only]
   |- rerank (LLM, capped 12 docs × 600 chars)
   |- answer (grounded, PII-sanitized)
   |- verify → end / retry(decompose) / abstain
```

## Project layout

```text
omdd-graphrag/
├── .env.example               # Configuration template
├── .gitignore                 # Excludes .env, data/, caches
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT
├── LoadLLM.py                 # LLM provider loader
├── indexing.py                # PDF → Chroma + BM25 JSONL
├── graph_store.py             # Kuzu schema + parameterized graph ops
├── graph_indexing.py          # LLM triple extraction → Kuzu
├── retrieving.py              # LangGraph adaptive RAG + GraphRAG agent
├── docs/                      # Source PDFs (user-provided, gitignored)
│   ├── HKEX_OMD_Derivatives_Binary_Interface_Specifications_v1-48.pdf
│   ├── HKEX-OMDD-Developers-Guide-v1.27.pdf
│   └── OMD_Connectivity_Guide_Derivatives-Market-v1.23.pdf
├── data/                      # Runtime artifacts (gitignored)
│   ├── chroma_omdd_db/
│   ├── bm25_docs_cache.jsonl
│   ├── graph/
│   └── graph_triples.jsonl
└── evaluation/
    ├── evaluate.py            # Benchmark runner
    ├── test_cases.json        # 8 test cases across 4 categories
    └── results.json           # Generated after evaluation
```

## Requirements

- Python 3.11+ (tested on 3.13.7)
- NVIDIA NIM API key (free tier available at build.nvidia.com)
- ~2 GB disk for Chroma + Kuzu artifacts

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/omdd-graphrag.git
cd omdd-graphrag
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\Activate.ps1     # Windows PowerShell
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the environment template and fill in your NVIDIA API key:

```bash
cp .env.example .env
```

Edit `.env` and set:

```dotenv
NVIDIA_API_KEY=nvapi-your-key-here
```

5. Place the three HKEX OMDD source PDFs in `docs/`:

```text
docs/HKEX_OMD_Derivatives_Binary_Interface_Specifications_v1-48.pdf
docs/HKEX-OMDD-Developers-Guide-v1.27.pdf
docs/OMD_Connectivity_Guide_Derivatives-Market-v1.23.pdf
```

## Workflow: indexing to retrieval

### Step 1 — Build vector + BM25 index

```bash
python indexing.py
```

- Prompts to reset the existing Chroma store (answer `yes` for a fresh build).
- Reads all PDFs in `docs/`, splits into page-aware chunks, indexes into Chroma, and writes `data/bm25_docs_cache.jsonl`.
- Expected output: `Success: indexing complete.`

### Step 2 — Build the Kuzu evidence graph

```bash
python graph_indexing.py
```

- Reads the BM25 cache produced in Step 1.
- Uses `GRAPH_EXTRACTION_MODEL` to extract evidence-backed triples per chunk.
- Stores triples in Kuzu (`data/graph/omdd.kuzu`) and writes an audit copy to `data/graph_triples.jsonl`.
- Expected output: `Graph indexing complete. Total triples extracted: N`

### Step 3 — Run the retrieval agent

```bash
python retrieving.py
```

Interactive prompt:

```text
Question (or 'exit'): What is the PacketHeader size?
Question (or 'exit'): How can I handle a message gap?
Question (or 'exit'): How to recover a message gap via the retransmission server?
Question (or 'exit'): How do I configure the OMD D-Lite connection?
Question (or 'exit'): Explain the end-to-end process for recovering a message gap,
                      including message semantics, recovery procedure, and connectivity requirements.
Question (or 'exit'): exit
```

Cross-document questions automatically trigger the Kuzu graph expansion route.

### Step 4 — Run the evaluation suite

```bash
python evaluation/evaluate.py
```

- Runs all 8 test cases in `evaluation/test_cases.json`.
- Writes `evaluation/results.json` with pass rate, source recall, latency, and graph-route accuracy.
- Compare `graph_used` against `expected_graph_route` to confirm the adaptive routing behaves correctly.

## Configuration

All tuning is via `.env` — no code changes needed:

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` | (required) | NVIDIA NIM API credential |
| `ANSWER_MODEL` | `qwen/qwen3.5-397b-a17b` | Answer generation |
| `JUDGE_MODEL` | `deepseek-ai/deepseek-v4-flash` | Grounding verification / retry decisions |
| `GRAPH_EXTRACTION_MODEL` | `qwen/qwen3.5-122b-a10b` | Triple extraction for Kuzu graph |
| `CONTEXT_WINDOWS_SIZE` | `32768` | LLM context window |
| `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | Chroma embedding model |
| `CHROMA_COLLECTION` | `omdd_collection_bge_large_v1` | Chroma collection name |
| `CHROMA_PERSIST_DIR` | `./data/chroma_omdd_db` | Chroma persistence path |
| `BM25_CACHE_PATH` | `./data/bm25_docs_cache.jsonl` | BM25 corpus cache |
| `KUZU_GRAPH_DIR` | `./data/graph/omdd.kuzu` | Kuzu database **file** path |
| `GRAPH_TRIPLES_PATH` | `./data/graph_triples.jsonl` | Triple audit log |
| `DENSE_K` / `BM25_K` | `20` / `20` | Per-retriever candidate count |
| `FINAL_K` | `36` | Merged candidate pool before rerank |
| `RERANK_TOP_N` | `8` | Chunks passed to answer LLM |
| `PER_SOURCE_K` | `5` | Per-source scoped dense pulls |
| `MAX_RETRIES` | `2` | Retry rounds before abstention |

## Security notes

- `.env` is gitignored — never commit API keys.
- The `NVIDIA_API_KEY` is read from the environment at runtime, not hardcoded.
- `sanitize_output()` redacts IPv4 addresses, emails, bearer tokens, and AWS keys from LLM output before returning answers.

## License

MIT — see `LICENSE`.

## Acknowledgements

- Built on LangGraph, LangChain, Chroma, and Kuzu.
- Source documents are HKEX OMD Derivatives Market technical manuals; they are user-provided and not distributed with this repository.
