"""
graph_indexing.py — Evidence-backed GraphRAG construction for HKEX OMDD

This script:
  1. Loads all chunks from the BM25 cache (JSONL) created by indexing.py.
  2. Uses an LLM to extract entities and relationships with provenance.
  3. Stores triples in Kuzu via graph_store.py.
  4. Writes a JSONL copy of extracted triples to disk for auditing.

Every relationship stores:
  - source: PDF filename
  - page: page number
  - chunk_id: chunk identifier
  - evidence_text: exact supporting sentence(s)
  - confidence: extraction confidence 0–1
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from LoadLLM import load_config, load_llm
from graph_store import open_graph


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def load_graph_index_config() -> dict:
    load_config()
    return {
        "bm25_cache_path": get_env("BM25_CACHE_PATH", "./bm25_docs_cache.jsonl"),
        "graph_triples_path": get_env("GRAPH_TRIPLES_PATH", "./data/graph_triples.jsonl"),
        "extraction_model": get_env("GRAPH_EXTRACTION_MODEL", "qwen/qwen3.5-397b-a17b"),
        "chunk_batch_size": int(get_env("GRAPH_CHUNK_BATCH", "40")),
    }


# ---------------------------------------------------------------------------
# LLM extraction schema
# ---------------------------------------------------------------------------

class Triple(BaseModel):
    subject: str = Field(description="Subject entity name")
    subject_type: str = Field(
        description="One of: Message, Field, Procedure, Component, NetworkResource"
    )
    predicate: str = Field(
        description="One of: CONTAINS_FIELD, HAS_MESSAGE_ID, TRIGGERED_BY, REQUIRES, CONNECTS_TO, CARRIES"
    )
    object: str = Field(description="Object entity name")
    object_type: str = Field(
        description="One of: Message, Field, Procedure, Component, NetworkResource"
    )
    evidence_text: str = Field(description="Exact supporting sentence(s) from the chunk")
    confidence: float = Field(description="Confidence 0–1")


class ChunkTriples(BaseModel):
    chunk_id: str
    source: str
    page: str
    triples: List[Triple] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# BM25 cache loader
# ---------------------------------------------------------------------------

def load_bm25_chunks(cache_path: str) -> List[Dict[str, Any]]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"BM25 cache not found at {cache_path}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = row.get("metadata", {})
            chunks.append({
                "chunk_id": meta.get("chunk_id", ""),
                "source": meta.get("source", ""),
                "page": str(meta.get("page", "n/a")),
                "text": row.get("page_content", ""),
            })
    return chunks


# ---------------------------------------------------------------------------
# LLM-based triple extraction
# ---------------------------------------------------------------------------

def extract_triples_from_chunk(
    model,
    chunk: Dict[str, Any],
) -> ChunkTriples:
    """
    Extract evidence-backed triples from a single chunk.
    """
    chunk_id = chunk["chunk_id"]
    source = chunk["source"]
    page = chunk["page"]
    text = chunk["text"]

    prompt = f"""
You are extracting an evidence-backed knowledge graph from HKEX OMDD technical manuals.

Available entity types:
- Message
- Field
- Procedure
- Component
- NetworkResource

Allowed relationship types:
- CONTAINS_FIELD (Message -> Field)
- HAS_MESSAGE_ID (Message -> Field)
- TRIGGERED_BY (Procedure -> Message)
- REQUIRES (Procedure -> Message|Field|Component)
- CONNECTS_TO (Component -> Component)
- CARRIES (NetworkResource -> Message)

Instructions:
1. Extract only relationships that are explicitly stated or strongly implied in the text.
2. For each relationship, copy the exact supporting sentence(s) into evidence_text.
3. Assign a confidence score between 0 and 1.
4. Do not invent entities or relationships.

Chunk metadata:
- chunk_id: {chunk_id}
- source: {source}
- page: {page}

Chunk text (truncated if long):
{text[:2000]}

Return ONLY valid JSON in this format:
{{
  "chunk_id": "{chunk_id}",
  "source": "{source}",
  "page": "{page}",
  "triples": [
    {{
      "subject": "...",
      "subject_type": "Message",
      "predicate": "CONTAINS_FIELD",
      "object": "...",
      "object_type": "Field",
      "evidence_text": "...",
      "confidence": 0.9
    }}
  ]
}}
"""
    res = model.invoke([{"role": "user", "content": prompt}])
    text_out = res.content if isinstance(res.content, str) else str(res.content)

    # Extract JSON block
    match = re.search(r"\{.*\}", text_out, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return ChunkTriples(**parsed)
        except Exception:
            pass

    # Fallback: empty triples
    return ChunkTriples(chunk_id=chunk_id, source=source, page=page, triples=[])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_graph_index_config()
    model = load_llm("Nvidia_API", config["extraction_model"])

    if model is None:
        raise RuntimeError("Failed to load graph extraction LLM.")

    print("--- HKEX OMDD Graph Indexer ---")
    print(f"BM25 cache path     : {config['bm25_cache_path']}")
    print(f"Graph triples output: {config['graph_triples_path']}")
    print(f"Extraction model    : {config['extraction_model']}")
    print(f"Chunk batch size    : {config['chunk_batch_size']}\n")

    chunks = load_bm25_chunks(config["bm25_cache_path"])
    print(f"Loaded {len(chunks)} chunks from BM25 cache.\n")

    graph = open_graph()
    triples_path = Path(config["graph_triples_path"])
    triples_path.parent.mkdir(parents=True, exist_ok=True)

    all_triples = []
    processed = 0

    with triples_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            chunk_triples = extract_triples_from_chunk(model, chunk)
            processed += 1

            for t in chunk_triples.triples:
                # Insert into Kuzu
                graph.add_edge(
                    rel_type=t.predicate,
                    from_label=t.subject_type,
                    from_name=t.subject,
                    to_label=t.object_type,
                    to_name=t.object,
                    source=chunk_triples.source,
                    page=chunk_triples.page,
                    chunk_id=chunk_triples.chunk_id,
                    evidence_text=t.evidence_text,
                    confidence=t.confidence,
                )

                # Record to JSONL
                row = {
                    "chunk_id": chunk_triples.chunk_id,
                    "source": chunk_triples.source,
                    "page": chunk_triples.page,
                    "subject": t.subject,
                    "subject_type": t.subject_type,
                    "predicate": t.predicate,
                    "object": t.object,
                    "object_type": t.object_type,
                    "evidence_text": t.evidence_text,
                    "confidence": t.confidence,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_triples.append(row)

            if processed % 10 == 0:
                print(f"Processed {processed}/{len(chunks)} chunks, "
                      f"extracted {len(all_triples)} triples so far.")

    graph.close()

    print(f"\nGraph indexing complete.")
    print(f"Total triples extracted: {len(all_triples)}")
    print(f"Triples saved to: {triples_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting.")