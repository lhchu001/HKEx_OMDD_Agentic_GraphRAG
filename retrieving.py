"""retrieving.py — Adaptive hybrid RAG + Kuzu GraphRAG for HKEX OMDD.

Run indexing.py, then graph_indexing.py, before running this program.
Graph expansion is used only for cross-document questions. Graph triples
are never directly supplied as answer evidence; their chunk_ids are used
to retrieve the original indexed chunks from Chroma.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

import kuzu
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langgraph.graph import END, START, StateGraph

from LoadLLM import load_config, load_llm, set_langsmith

SOURCE_BINARY_SPEC = "HKEX_OMD_Derivatives_Binary_Interface_Specifications_v1-48.pdf"
SOURCE_DEV_GUIDE = "HKEX-OMDD-Developers-Guide-v1.27.pdf"
SOURCE_CONNECTIVITY = "OMD_Connectivity_Guide_Derivatives-Market-v1.23.pdf"
SOURCE_PRIORITY = [SOURCE_BINARY_SPEC, SOURCE_DEV_GUIDE, SOURCE_CONNECTIVITY]


class RAGState(TypedDict):
    question: str
    question_type: str
    queries: List[Dict[str, str]]
    seed_entities: List[str]
    graph_chunk_ids: List[str]
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str
    retry_count: int


def get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def config() -> Dict[str, Any]:
    load_config()
    return {
        "embed_model": get_env("EMBED_MODEL", "BAAI/bge-large-en-v1.5"),
        "device": get_env("EMBED_DEVICE", "cpu"),
        "normalize": get_env("NORMALIZE_EMBEDDINGS", "true").lower() == "true",
        "collection": get_env("CHROMA_COLLECTION", "omdd_collection_bge_large_v1"),
        "chroma_dir": get_env("CHROMA_PERSIST_DIR", "./data/chroma_omdd_db"),
        "bm25_cache": get_env("BM25_CACHE_PATH", "./data/bm25_docs_cache.jsonl"),
        "graph_dir": get_env("KUZU_GRAPH_DIR", "./data/graph"),
        "dense_k": int(get_env("DENSE_K", "20")),
        "bm25_k": int(get_env("BM25_K", "20")),
        "final_k": int(get_env("FINAL_K", "36")),
        "rerank_top_n": int(get_env("RERANK_TOP_N", "8")),
        "per_source_k": int(get_env("PER_SOURCE_K", "5")),
        "max_retries": int(get_env("MAX_RETRIES", "2")),
        "answer_model": get_env("ANSWER_MODEL", "moonshotai/kimi-k2.6"),
        "judge_model": get_env("JUDGE_MODEL", "deepseek-ai/deepseek-v4-pro"),
    }


SETTINGS = config()
set_langsmith(get_env("LANGSMITH_PROJECT", "hkex_omdd_rag_kuzu"))
ANSWER_LLM = load_llm("Nvidia_API", SETTINGS["answer_model"])
JUDGE_LLM = load_llm("Nvidia_API", SETTINGS["judge_model"])
if ANSWER_LLM is None or JUDGE_LLM is None:
    raise RuntimeError("Could not initialize answer or judge LLM.")

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=SETTINGS["embed_model"],
    model_kwargs={"device": SETTINGS["device"], "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": SETTINGS["normalize"]},
)
VECTOR_STORE = Chroma(
    collection_name=SETTINGS["collection"],
    embedding_function=EMBEDDINGS,
    persist_directory=SETTINGS["chroma_dir"],
)


def llm_text(model, prompt: str, fallback: str = "") -> str:
    try:
        response = model.invoke([{"role": "user", "content": prompt}])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        print(f"[Warning] LLM call failed: {exc}")
        return fallback


def parse_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text.strip())
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def doc_key(doc: Document) -> str:
    content = doc.page_content[:1000].encode("utf-8")
    return "|".join([
        str(doc.metadata.get("source", "")),
        str(doc.metadata.get("page", "")),
        str(doc.metadata.get("chunk_id", "")),
        hashlib.md5(content).hexdigest(),
    ])


def dedupe_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    result = []
    for doc in docs:
        key = doc_key(doc)
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


def sanitize_output(text: str) -> str:
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED_IP]", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", text)
    text = re.sub(r"(?i)Bearer\s+[A-Za-z0-9\-._~+/]+", "Bearer [REDACTED_TOKEN]", text)
    return text


def load_bm25_documents(path_str: str) -> List[Document]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"BM25 cache is missing: {path}")
    docs = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(Document(
                page_content=row["page_content"],
                metadata=row.get("metadata", {}),
            ))
    return docs


BM25_DOCS = load_bm25_documents(SETTINGS["bm25_cache"])
BM25 = BM25Retriever.from_documents(BM25_DOCS)
BM25.k = SETTINGS["bm25_k"]
DENSE = VECTOR_STORE.as_retriever(search_kwargs={"k": SETTINGS["dense_k"]})
HYBRID = EnsembleRetriever(retrievers=[BM25, DENSE], weights=[0.45, 0.55])


def source_targets(question_type: str) -> List[str]:
    if question_type == "message_format":
        return [SOURCE_BINARY_SPEC]
    if question_type == "operational_procedure":
        return [SOURCE_DEV_GUIDE, SOURCE_BINARY_SPEC]
    if question_type == "connectivity":
        return [SOURCE_CONNECTIVITY, SOURCE_DEV_GUIDE]
    return SOURCE_PRIORITY


def canonical_source(value: str) -> str | None:
    low = value.lower()
    if not value or low == "all":
        return None
    aliases = {
        "binary": SOURCE_BINARY_SPEC,
        "interface": SOURCE_BINARY_SPEC,
        "specification": SOURCE_BINARY_SPEC,
        "developer": SOURCE_DEV_GUIDE,
        "procedure": SOURCE_DEV_GUIDE,
        "connectivity": SOURCE_CONNECTIVITY,
        "d-lite": SOURCE_CONNECTIVITY,
    }
    for token, source in aliases.items():
        if token in low:
            return source
    for source in SOURCE_PRIORITY:
        if value.lower() in source.lower() or source.lower() in value.lower():
            return source
    return None


def classify_question(state: RAGState) -> Dict[str, Any]:
    prompt = f"""Classify this HKEX OMDD question.

Question: {state['question']}

Return JSON only:
{{"question_type":"message_format|operational_procedure|connectivity|cross_document"}}

Definitions:
- message_format: exact message, field, type, encoding, length or packet structure
- operational_procedure: recovery, sequence, gap, login, operational processing
- connectivity: network, multicast, channel, D-Lite, host or port configuration
- cross_document: answer requires two or more manuals, or classification is uncertain
"""
    value = parse_json(llm_text(ANSWER_LLM, prompt, '{"question_type":"cross_document"}'))
    qtype = value.get("question_type", "cross_document")
    if qtype not in {"message_format", "operational_procedure", "connectivity", "cross_document"}:
        qtype = "cross_document"
    return {"question_type": qtype}


def plan_queries(state: RAGState) -> Dict[str, Any]:
    prompt = f"""You are a retrieval planner for three HKEX OMDD manuals.

Question type: {state['question_type']}
Question: {state['question']}

Available source labels:
- {SOURCE_BINARY_SPEC}: messages, fields, binary layout
- {SOURCE_DEV_GUIDE}: procedures, recovery, operational workflow
- {SOURCE_CONNECTIVITY}: network, multicast, channels, OMD D-Lite
- ALL: genuinely cross-document search

Generate 3 to 5 retrieval queries. Use exact protocol names, message names,
field names, acronyms, and recovery or connectivity terms where applicable.

Return JSON only:
{{"queries":[{{"search_term":"...","target_document":"exact source name or ALL"}}]}}
"""
    value = parse_json(llm_text(ANSWER_LLM, prompt, '{"queries":[]}'))
    raw = value.get("queries", [])
    queries = []
    for item in raw:
        if isinstance(item, dict) and item.get("search_term", "").strip():
            queries.append({
                "search_term": item["search_term"].strip(),
                "target_document": item.get("target_document", "ALL").strip(),
            })
    if not queries:
        queries = [{"search_term": state["question"], "target_document": "ALL"}]
    return {"queries": queries[:5]}


def retrieve_hybrid(state: RAGState) -> Dict[str, Any]:
    candidates: List[Document] = []
    type_targets = source_targets(state["question_type"])
    for query in state["queries"]:
        term = query["search_term"]
        target = canonical_source(query["target_document"])
        raw_docs = HYBRID.invoke(term)
        if target:
            matched = [d for d in raw_docs if d.metadata.get("source") == target]
            candidates.extend(matched or raw_docs)
        else:
            candidates.extend(raw_docs)
        for source in type_targets:
            try:
                scoped = VECTOR_STORE.as_retriever(search_kwargs={
                    "k": SETTINGS["per_source_k"],
                    "filter": {"source": source},
                })
                candidates.extend(scoped.invoke(term))
            except Exception as exc:
                print(f"[Warning] Source-scoped search failed: {exc}")
    return {"retrieved_docs": dedupe_docs(candidates)[:SETTINGS["final_k"]]}


def graph_seed_entities(state: RAGState) -> Dict[str, Any]:
    if state["question_type"] != "cross_document":
        return {"seed_entities": [], "graph_chunk_ids": []}
    prompt = f"""Extract up to five exact entity names likely useful for a Kuzu knowledge-graph traversal.
Entities can be HKEX OMDD messages, fields, procedures, components, or network resources.
Do not invent entities.

Question: {state['question']}

Return JSON only: {{"entities":["..."]}}
"""
    value = parse_json(llm_text(ANSWER_LLM, prompt, '{"entities":[]}'))
    entities = [str(v).strip() for v in value.get("entities", []) if str(v).strip()]
    return {"seed_entities": entities[:5], "graph_chunk_ids": []}


def graph_expand(state: RAGState) -> Dict[str, Any]:
    if state["question_type"] != "cross_document" or not state["seed_entities"]:
        return {"graph_chunk_ids": []}

    graph_file = Path(SETTINGS["graph_dir"])
    # Check that either the database file exists or its parent directory is valid
    if not graph_file.exists() and not graph_file.parent.exists():
        print("[Warning] Kuzu graph file is unavailable; continuing with hybrid retrieval.")
        return {"graph_chunk_ids": []}

    chunk_ids = set()
    try:
        db = kuzu.Database(str(graph_file))
        conn = kuzu.Connection(db)
        for entity in state["seed_entities"]:
            escaped = entity.replace("'", "''")
            cypher = f"""
            MATCH (a)-[r*1..2]-(b)
            WHERE lower(a.name) = lower('{escaped}') OR lower(b.name) = lower('{escaped}')
            UNWIND r AS edge
            RETURN DISTINCT edge.chunk_id AS chunk_id
            """
            result = conn.execute(cypher)
            while result.has_next():
                row = result.get_next()
                if row[0]:
                    chunk_ids.add(str(row[0]))
        conn.close()
    except Exception as exc:
        print(f"[Warning] Kuzu traversal failed: {exc}")
    return {"graph_chunk_ids": list(chunk_ids)}


def fetch_graph_evidence(state: RAGState) -> Dict[str, Any]:
    if not state["graph_chunk_ids"]:
        return {}
    wanted = set(state["graph_chunk_ids"])
    graph_docs = [doc for doc in BM25_DOCS if doc.metadata.get("chunk_id") in wanted]
    merged = dedupe_docs(state["retrieved_docs"] + graph_docs)
    return {"retrieved_docs": merged[:SETTINGS["final_k"]]}


def rerank(state: RAGState) -> Dict[str, Any]:
    docs = state["retrieved_docs"]
    if not docs:
        return {"reranked_docs": []}
    pool = docs[: min(12, len(docs))]
    prompt_docs = "\n\n".join(
        f"[{i}] source={doc.metadata.get('source')} page={doc.metadata.get('page')}\n{doc.page_content[:600]}"
        for i, doc in enumerate(pool)
    )
    prompt = f"""Rank passages for the user question.

Question: {state['question']}

Return only numeric indices in descending relevance order, for example: [2] [0] [4]

Passages:
{prompt_docs}
"""
    text = llm_text(JUDGE_LLM, prompt, "")
    order = []
    for token in re.findall(r"\d+", text):
        index = int(token)
        if 0 <= index < len(pool) and index not in order:
            order.append(index)
    ranked = [pool[i] for i in order]
    ranked.extend(doc for i, doc in enumerate(pool) if i not in order)
    ranked.extend(docs[len(pool):])
    return {"reranked_docs": ranked[:SETTINGS["rerank_top_n"]]}


def answer(state: RAGState) -> Dict[str, Any]:
    context = "\n\n".join(
        f"[{i + 1}] Source={doc.metadata.get('source')} | Page={doc.metadata.get('page')} | Chunk={doc.metadata.get('chunk_id')}\n{doc.page_content}"
        for i, doc in enumerate(state["reranked_docs"])
    )
    prompt = f"""You are an HKEX OMDD technical assistant.
Use only the supplied original document passages. The graph may have helped find passages, but graph facts are not independently authoritative.
If the passages do not support the answer, respond exactly: Do not know.

Question:
{state['question']}

Evidence:
{context}

Requirements:
- Answer precisely and concisely.
- Combine sources only when evidence supports the connection.
- Cite each factual statement using [Source filename, p. page].
- Do not disclose IP addresses, credentials, tokens, or private network details.
"""
    text = llm_text(ANSWER_LLM, prompt, "Do not know").strip() or "Do not know"
    return {"answer": sanitize_output(text)}


def verify(state: RAGState) -> Literal["end", "retry", "abstain"]:
    context = "\n\n".join(doc.page_content[:900] for doc in state["reranked_docs"])
    prompt = f"""Assess whether this answer is fully supported by the supplied evidence.

Question: {state['question']}
Answer: {state['answer']}
Evidence: {context}

Return JSON only:
{{"grounded":true,"answerable":true,"needs_retry":false}}
"""
    value = parse_json(llm_text(JUDGE_LLM, prompt, '{"grounded":false,"answerable":false,"needs_retry":false}'))
    if value.get("grounded") and value.get("answerable"):
        return "end"
    if state["retry_count"] < SETTINGS["max_retries"] and value.get("needs_retry"):
        return "retry"
    return "abstain"


def decompose(state: RAGState) -> Dict[str, Any]:
    prompt = f"""The following HKEX OMDD question was not adequately supported.
Question: {state['question']}
Current answer: {state['answer']}
Generate two focused retrieval formulations. Return JSON only:
{{"queries":[{{"search_term":"...","target_document":"ALL"}}]}}
"""
    value = parse_json(llm_text(ANSWER_LLM, prompt, '{"queries":[]}'))
    queries = value.get("queries", [])
    clean = [q for q in queries if isinstance(q, dict) and q.get("search_term")]
    if not clean:
        clean = [{"search_term": state["question"], "target_document": "ALL"}]
    return {
        "queries": clean[:3],
        "seed_entities": [],
        "graph_chunk_ids": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "retry_count": state["retry_count"] + 1,
    }


def abstain(_: RAGState) -> Dict[str, Any]:
    return {"answer": "Do not know"}


def build_app():
    graph = StateGraph(RAGState)
    graph.add_node("classify_question", classify_question)
    graph.add_node("plan_queries", plan_queries)
    graph.add_node("retrieve_hybrid", retrieve_hybrid)
    graph.add_node("graph_seed_entities", graph_seed_entities)
    graph.add_node("graph_expand", graph_expand)
    graph.add_node("fetch_graph_evidence", fetch_graph_evidence)
    graph.add_node("rerank", rerank)
    graph.add_node("answer", answer)
    graph.add_node("decompose", decompose)
    graph.add_node("abstain", abstain)
    graph.add_edge(START, "classify_question")
    graph.add_edge("classify_question", "plan_queries")
    graph.add_edge("plan_queries", "retrieve_hybrid")
    graph.add_edge("retrieve_hybrid", "graph_seed_entities")
    graph.add_edge("graph_seed_entities", "graph_expand")
    graph.add_edge("graph_expand", "fetch_graph_evidence")
    graph.add_edge("fetch_graph_evidence", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_conditional_edges("answer", verify, {"end": END, "retry": "decompose", "abstain": "abstain"})
    graph.add_edge("decompose", "retrieve_hybrid")
    graph.add_edge("abstain", END)
    return graph.compile()


APP = build_app()


def ask(question: str) -> Dict[str, Any]:
    return APP.invoke({
        "question": question,
        "question_type": "cross_document",
        "queries": [],
        "seed_entities": [],
        "graph_chunk_ids": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "retry_count": 0,
    })


def main() -> None:
    print("--- HKEX OMDD Adaptive Hybrid + Kuzu GraphRAG ---")
    while True:
        question = input("Question (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        if not question:
            continue
        result = ask(question)
        print("\nAnswer\n------")
        print(result["answer"])
        print("\nRetrieved sources\n-----------------")
        for doc in result["reranked_docs"]:
            print(f"- {doc.metadata.get('source')} p.{doc.metadata.get('page')} ({doc.metadata.get('chunk_id')})")
        print()


if __name__ == "__main__":
    main()