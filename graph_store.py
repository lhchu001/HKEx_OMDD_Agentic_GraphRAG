"""
graph_store.py — Kuzu-backed evidence graph for HKEX OMDD (v0.11+ compatible)

Schema:
  - Entity types: Message, Field, Procedure, Component, NetworkResource, DocumentChunk
  - Relationships (all with provenance properties):
      (Message)-[:CONTAINS_FIELD]->(Field)
      (Message)-[:HAS_MESSAGE_ID]->(Field)
      (Procedure)-[:TRIGGERED_BY]->(Message)
      (Procedure)-[:REQUIRES]->(Message | Field | Component)
      (Component)-[:CONNECTS_TO]->(Component)
      (NetworkResource)-[:CARRIES]->(Message)

Notes:
  - Kuzu does NOT support "ANY" as a relationship endpoint wildcard.
    Multi-pair relationships (e.g. REQUIRES) use multiple FROM...TO clauses.
  - All data values are passed via parameterized queries ($params), never
    string interpolation, so evidence text containing single quotes,
    backslashes or newlines can never break Cypher parsing.

Every edge stores provenance:
  - source, page, chunk_id, evidence_text, confidence
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import kuzu


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def load_graph_config() -> dict:
    return {
        "graph_dir": get_env("KUZU_GRAPH_DIR", "./data/graph/omdd.kuzu"),
        "triples_path": get_env("GRAPH_TRIPLES_PATH", "./data/graph_triples.jsonl"),
    }


# Valid node labels used to guard query construction
VALID_NODE_LABELS = {
    "Message", "Field", "Procedure", "Component", "NetworkResource", "DocumentChunk"
}

# Valid relationship types -> allowed (from_label, to_label) endpoint pairs
VALID_REL_ENDPOINTS = {
    "CONTAINS_FIELD":  {("Message", "Field")},
    "HAS_MESSAGE_ID":  {("Message", "Field")},
    "TRIGGERED_BY":    {("Procedure", "Message")},
    "REQUIRES":        {("Procedure", "Message"), ("Procedure", "Field"), ("Procedure", "Component")},
    "CONNECTS_TO":     {("Component", "Component")},
    "CARRIES":         {("NetworkResource", "Message")},
}


# ---------------------------------------------------------------------------
# Graph wrapper
# ---------------------------------------------------------------------------

class OMDDGraph:
    def __init__(self, graph_path: str):
        self.graph_path = Path(graph_path)
        # Ensure only the parent folder exists (e.g. ./data/graph)
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = kuzu.Database(str(self.graph_path))
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self) -> None:
        # Node tables
        self._run("CREATE NODE TABLE IF NOT EXISTS Message(name STRING, PRIMARY KEY(name))")
        self._run("CREATE NODE TABLE IF NOT EXISTS Field(name STRING, PRIMARY KEY(name))")
        self._run("CREATE NODE TABLE IF NOT EXISTS Procedure(name STRING, PRIMARY KEY(name))")
        self._run("CREATE NODE TABLE IF NOT EXISTS Component(name STRING, PRIMARY KEY(name))")
        self._run("CREATE NODE TABLE IF NOT EXISTS NetworkResource(name STRING, PRIMARY KEY(name))")
        self._run("CREATE NODE TABLE IF NOT EXISTS DocumentChunk(chunk_id STRING, source STRING, page STRING, PRIMARY KEY(chunk_id))")

        # Relationship tables with provenance.
        # No "ANY" wildcard: REQUIRES uses multiple concrete FROM/TO pairs.
        rel_props = (
            "source STRING, page STRING, chunk_id STRING, evidence_text STRING, confidence DOUBLE"
        )
        self._run(f"CREATE REL TABLE IF NOT EXISTS CONTAINS_FIELD(FROM Message TO Field, {rel_props})")
        self._run(f"CREATE REL TABLE IF NOT EXISTS HAS_MESSAGE_ID(FROM Message TO Field, {rel_props})")
        self._run(f"CREATE REL TABLE IF NOT EXISTS TRIGGERED_BY(FROM Procedure TO Message, {rel_props})")
        self._run(
            f"CREATE REL TABLE IF NOT EXISTS REQUIRES("
            f"FROM Procedure TO Message, "
            f"FROM Procedure TO Field, "
            f"FROM Procedure TO Component, {rel_props})"
        )
        self._run(f"CREATE REL TABLE IF NOT EXISTS CONNECTS_TO(FROM Component TO Component, {rel_props})")
        self._run(f"CREATE REL TABLE IF NOT EXISTS CARRIES(FROM NetworkResource TO Message, {rel_props})")

    def _run(self, query_str: str, parameters: Optional[dict] = None):
        """
        Execute a Cypher query via the Kuzu Python API.
        Accepts an optional parameters dict for $-style placeholders.
        Kuzu cannot bind None, so None values are coerced to empty strings.
        """
        if parameters is None:
            return self.conn.execute(query_str)
        params = {
            k: (v if v is not None else "")
            for k, v in parameters.items()
        }
        return self.conn.execute(query_str, parameters=params)

    def upsert_node(self, label: str, name: str) -> None:
        if label not in VALID_NODE_LABELS:
            return
        clean_name = str(name).strip()
        if not clean_name:
            return
        # Label is allowlisted; the name value is bound as a parameter
        query_str = f"MERGE (n:{label} {{name: $name}})"
        try:
            self._run(query_str, {"name": clean_name})
        except Exception:
            pass

    def upsert_chunk_node(self, chunk_id: str, source: str, page: str) -> None:
        clean_cid = str(chunk_id).strip()
        if not clean_cid:
            return
        query_str = (
            "MERGE (c:DocumentChunk {chunk_id: $chunk_id}) "
            "ON CREATE SET c.source = $source, c.page = $page"
        )
        try:
            self._run(query_str, {
                "chunk_id": clean_cid,
                "source": str(source or ""),
                "page": str(page or ""),
            })
        except Exception:
            pass

    def add_edge(
        self,
        rel_type: str,
        from_label: str,
        from_name: str,
        to_label: str,
        to_name: str,
        source: str,
        page: str,
        chunk_id: str,
        evidence_text: str,
        confidence: float,
    ) -> None:
        # Validate structural parts against allowlists
        if rel_type not in VALID_REL_ENDPOINTS:
            return
        if (from_label, to_label) not in VALID_REL_ENDPOINTS[rel_type]:
            return

        self.upsert_node(from_label, from_name)
        self.upsert_node(to_label, to_name)
        self.upsert_chunk_node(chunk_id, source, page)

        # Labels and rel type are allowlisted identifiers -> safe to interpolate.
        # All string VALUES are bound as $parameters -> immune to quote issues.
        query_str = f"""
        MATCH (a:{from_label} {{name: $from_name}}), (b:{to_label} {{name: $to_name}})
        CREATE (a)-[r:{rel_type} {{
            source: $source,
            page: $page,
            chunk_id: $chunk_id,
            evidence_text: $evidence,
            confidence: $confidence
        }}]->(b)
        """
        try:
            self._run(query_str, {
                "from_name": str(from_name or "").strip(),
                "to_name": str(to_name or "").strip(),
                "source": str(source or ""),
                "page": str(page or ""),
                "chunk_id": str(chunk_id or ""),
                "evidence": str(evidence_text or "").strip(),
                "confidence": float(confidence),
            })
        except Exception as exc:
            print(f"[graph_store] Edge insert failed: {exc}")

    def traverse_from_entity(
        self,
        label: str,
        name: str,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        if label not in VALID_NODE_LABELS:
            return []
        clean_name = str(name).strip()
        query_str = f"""
        MATCH (start:{label} {{name: $name}})-[r*1..{max_hops}]-(neighbor)
        UNWIND r AS edge
        RETURN DISTINCT edge.chunk_id AS chunk_id
        """
        results = []
        try:
            res = self._run(query_str, {"name": clean_name})
            while res.has_next():
                row = res.get_next()
                if row and row[0]:
                    results.append({"chunk_id": str(row[0])})
        except Exception as exc:
            print(f"[graph_store] Traversal failed: {exc}")
        return results

    def find_evidence_chunks_for_path(self, path_rows: List[Dict]) -> List[str]:
        return list({row["chunk_id"] for row in path_rows if row.get("chunk_id")})

    def close(self) -> None:
        self.conn.close()
        self.db.close()


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def open_graph() -> OMDDGraph:
    cfg = load_graph_config()
    return OMDDGraph(cfg["graph_dir"])
