# HKEX OMDD Agentic RAG with Kuzu GraphRAG — Architecture

## Goals

- Answer HKEX OMDD questions using three manuals:
  - Binary Interface Specification
  - Developers Guide
  - Connectivity Guide
- Use GraphRAG only when a question genuinely requires multi-document reasoning.
- Preserve strict grounding: answers must be supported by original document chunks, not graph triples alone.

## Components

1. `indexing.py`
   - Converts PDFs to Markdown pages.
   - Splits into chunks with page and section metadata.
   - Indexes chunks in Chroma.
   - Writes BM25 JSONL cache.

2. `graph_indexing.py`
   - Reads BM25 cache.
   - Extracts evidence-backed triples using an LLM.
   - Stores triples in Kuzu with provenance (source, page, chunk_id, evidence_text, confidence).

3. `graph_store.py`
   - Kuzu schema and traversal helpers.
   - Node types: Message, Field, Procedure, Component, NetworkResource, DocumentChunk.
   - Relationship types: CONTAINS_FIELD, HAS_MESSAGE_ID, TRIGGERED_BY, REQUIRES, CONNECTS_TO, CARRIES, DEFINES, CITED_IN.

4. `retrieving.py`
   - LangGraph workflow:
     - classify_question
     - plan_queries
     - retrieve_hybrid
     - graph_seed_entities (cross-document only)
     - graph_expand (Kuzu traversal)
     - fetch_graph_evidence (join graph chunk_ids to BM25 chunks)
     - rerank
     - answer
     - verify (groundedness check)
     - decompose (retry loop)
   - Hybrid retrieval is always used; graph expansion is conditional.

5. `evaluation/evaluate.py` + `test_cases.json`
   - Offline benchmark suite covering:
     - Exact protocol lookups
     - Procedures
     - Connectivity
     - Cross-document multi-hop questions
   - Metrics: pass rate, source recall, latency, graph-route accuracy.

## Data Flow

```text
docs/*.pdf
   |
   v
indexing.py
   |- Chroma vectors
   |- BM25 JSONL
   |
   v
graph_indexing.py
   |- Kuzu graph with provenance
   |
   v
retrieving.py
   |- classify_question
   |- plan_queries
   |- retrieve_hybrid
   |- (if cross_document) graph_seed_entities → graph_expand → fetch_graph_evidence
   |- rerank
   |- answer
   |- verify → end / retry / abstain
```

## Design Decisions

- GraphRAG is an evidence expander, not an answer source. The answer LLM sees only original chunks.
- Kuzu is embedded; no external database server is required.
- Graph expansion is triggered only for `cross_document` questions to avoid unnecessary latency on exact lookups.
- Reranking is applied after merging hybrid and graph-sourced chunks to ensure the most relevant evidence is used.

## Test Procedure

1. Place PDFs in `docs/`.
2. Run `python indexing.py` and confirm success.
3. Run `python graph_indexing.py` and confirm graph creation.
4. Run `python retrieving.py` and manually test sample questions.
5. Run `python evaluation/evaluate.py` and inspect `evaluation/results.json`.
6. Compare pass rates and latencies for cross-document vs non-cross-document subsets.