"""
indexing.py — HKEX OMDD vector + BM25 index builder

This script:
  1. Converts each PDF in docs/ into page-level Markdown using pymupdf4llm.
  2. Splits each page into chunks while preserving page and section metadata.
  3. Indexes all chunks into Chroma using BGE embeddings.
  4. Writes a JSONL corpus for BM25 and downstream GraphRAG indexing.

Metadata preserved per chunk:
  - source: PDF filename
  - page: original page number
  - page_label: string version of page
  - chunk_id: unique identifier "<filename>::chunk_<N>"
  - document_type: "HKEX_OMDD_PDF"
  - section: breadcrumb from Markdown header hierarchy (optional)
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import List

import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from LoadLLM import load_config


# ---------------------------------------------------------------------------
# Known HKEX OMDD source filenames (must match retrieving.py SOURCE_* constants)
# ---------------------------------------------------------------------------

KNOWN_SOURCES = {
    "HKEX_OMD_Derivatives_Binary_Interface_Specifications_v1-48.pdf",
    "HKEX-OMDD-Developers-Guide-v1.27.pdf",
    "OMD_Connectivity_Guide_Derivatives-Market-v1.23.pdf",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def load_index_config() -> dict:
    load_config()
    return {
        "embed_model":          get_env("EMBED_MODEL",          "BAAI/bge-large-en-v1.5"),
        "device":               get_env("EMBED_DEVICE",         "cpu"),
        "normalize_embeddings": get_env("NORMALIZE_EMBEDDINGS", "true").lower() == "true",
        "collection_name":      get_env("CHROMA_COLLECTION",    "omdd_collection_bge_large_v1"),
        "persist_directory":    get_env("CHROMA_PERSIST_DIR",   "./data/chroma_omdd_db"),
        "bm25_cache_path":      get_env("BM25_CACHE_PATH",      "./data/bm25_docs_cache.jsonl"),
        "chunk_size":           int(get_env("CHUNK_SIZE",       "1200")),
        "chunk_overlap":        int(get_env("CHUNK_OVERLAP",    "200")),
    }


def parse_input_paths(user_input: str) -> List[str]:
    file_paths    = re.findall(r'(?:[^\s"\']+|"[^"]*"|\'[^\']*\')+', user_input)
    cleaned_paths = [p.strip("\"'") for p in file_paths]
    return [p for p in cleaned_paths if os.path.exists(p)]


def reset_vector_store_dir(persist_directory: str) -> None:
    path = Path(persist_directory)
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed existing vector store directory: {persist_directory}")


def build_embeddings(model_name: str, device: str,
                     normalize_embeddings: bool) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )


def create_vector_store(embeddings, collection_name: str,
                        persist_directory: str) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


# ---------------------------------------------------------------------------
# PDF → page-level Documents
# ---------------------------------------------------------------------------

def md_doc_conversion(file_path: str) -> List[Document]:
    """
    Convert a PDF into one LangChain Document per page using pymupdf4llm.
    Preserves the original page number in metadata['page'].
    """
    try:
        pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    except Exception as exc:
        print(f"  [Error] pymupdf4llm failed on {file_path}: {exc}")
        return []

    docs: List[Document] = []
    basename = os.path.basename(file_path)

    for page_num, page in enumerate(pages, start=1):
        meta = dict(page.get("metadata", {}) or {})

        raw_page              = meta.get("page", page_num)
        meta["page"]          = raw_page
        meta["page_label"]    = str(raw_page)
        meta["source"]        = basename
        meta["file_path"]     = str(Path(file_path).resolve())
        meta["document_type"] = "HKEX_OMDD_PDF"

        if basename not in KNOWN_SOURCES:
            meta["unknown_source"] = True

        content = page.get("text", "").strip()
        if content:
            docs.append(Document(page_content=content, metadata=meta))

    return docs


# ---------------------------------------------------------------------------
# Chunk-level splitting with page propagation
# ---------------------------------------------------------------------------

def split_documents(
    docs: List[Document],
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Two-stage splitting per page:
      1. MarkdownHeaderTextSplitter  – honours section boundaries
      2. RecursiveCharacterTextSplitter – caps individual chunk size

    Page numbers are propagated into every chunk so retrieving.py
    can cite correct pages. Each page is processed individually so
    its page number is never lost during splitting.
    """
    basename  = os.path.basename(file_path)
    abs_path  = str(Path(file_path).resolve())

    headers_to_split_on = [
        ("#",    "Header 1"),
        ("##",   "Header 2"),
        ("###",  "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_split_docs: List[Document] = []
    chunk_counter = 0

    for page_doc in docs:
        page_num = page_doc.metadata.get("page", "n/a")
        content  = page_doc.page_content

        if not content.strip():
            continue

        try:
            md_header_splits = markdown_splitter.split_text(content)
        except Exception:
            md_header_splits = [Document(page_content=content)]

        for hdr_doc in md_header_splits:
            char_splits = text_splitter.split_documents([hdr_doc])

            for split_doc in char_splits:
                if not split_doc.page_content.strip():
                    continue

                chunk_counter += 1
                meta = dict(split_doc.metadata)

                # Propagate page number from parent page document
                if "page" not in meta or meta.get("page") in (None, "n/a", ""):
                    meta["page"] = page_num
                meta["page_label"] = str(meta["page"])

                meta["source"]        = basename
                meta["file_path"]     = abs_path
                meta["chunk_id"]      = f"{basename}::chunk_{chunk_counter}"
                meta["document_type"] = "HKEX_OMDD_PDF"

                # Section breadcrumb from markdown header hierarchy
                breadcrumb_parts = []
                for hdr_key in ("Header 1", "Header 2", "Header 3", "Header 4"):
                    if hdr_val := meta.get(hdr_key, ""):
                        breadcrumb_parts.append(hdr_val)
                if breadcrumb_parts:
                    meta["section"] = " > ".join(breadcrumb_parts)

                all_split_docs.append(
                    Document(page_content=split_doc.page_content, metadata=meta)
                )

    return all_split_docs


# ---------------------------------------------------------------------------
# BM25 cache writer (append-safe, de-duplicated)
# ---------------------------------------------------------------------------

def _load_existing_keys(cache_path: str) -> set:
    path = Path(cache_path)
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row      = json.loads(line)
                chunk_id = row.get("metadata", {}).get("chunk_id", "")
                if chunk_id:
                    keys.add(chunk_id)
            except Exception:
                pass
    return keys


def write_bm25_cache(docs: List[Document], bm25_cache_path: str,
                     append: bool = False) -> None:
    """
    append=True  → add only new chunks (safe for incremental indexing)
    append=False → overwrite the file (full re-index)
    """
    path = Path(bm25_cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append:
        existing_keys = _load_existing_keys(bm25_cache_path)
        new_docs      = [d for d in docs
                         if d.metadata.get("chunk_id", "") not in existing_keys]
        mode = "a"
    else:
        new_docs = docs
        mode     = "w"

    if not new_docs:
        print("  BM25 cache: no new chunks to write.")
        return

    with path.open(mode, encoding="utf-8") as f:
        for doc in new_docs:
            f.write(
                json.dumps(
                    {"page_content": doc.page_content, "metadata": doc.metadata},
                    ensure_ascii=False,
                ) + "\n"
            )

    verb = "appended" if append else "written"
    print(f"  BM25 cache {verb}: {len(new_docs)} chunks → {bm25_cache_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = load_index_config()

    print("--- HKEX OMDD Vector Database Indexer ---")
    print("Type 'quit', 'exit', 'ctrl c', or 'ctrl q' to stop.")
    print(f"Embedding model   : {config['embed_model']}")
    print(f"Collection name   : {config['collection_name']}")
    print(f"Persist directory : {config['persist_directory']}")
    print(f"BM25 cache path   : {config['bm25_cache_path']}")
    print(f"Chunk size        : {config['chunk_size']}")
    print(f"Chunk overlap     : {config['chunk_overlap']}\n")

    reset_choice = input("Reset existing vector store before indexing? (yes/no): ").strip().lower()
    do_reset     = reset_choice in {"yes", "y"}
    if do_reset:
        reset_vector_store_dir(config["persist_directory"])

    embeddings   = build_embeddings(
        model_name=config["embed_model"],
        device=config["device"],
        normalize_embeddings=config["normalize_embeddings"],
    )
    vector_store = create_vector_store(
        embeddings=embeddings,
        collection_name=config["collection_name"],
        persist_directory=config["persist_directory"],
    )

    while True:
        user_input = input("\nInput the list of documents to index: ").strip()

        if user_input.lower() in {"quit", "exit", "ctrl c", "ctrl q"}:
            print("Exiting. Goodbye!")
            break

        if not user_input:
            continue

        valid_paths = parse_input_paths(user_input)
        
        if not valid_paths:
            print("No valid file paths detected. Please verify your inputs.")
            continue

        combined_split_docs: List[Document] = []

        for file_path in valid_paths:
            basename = os.path.basename(file_path)
            print(f"\n-> Loading: {file_path}")

            if basename not in KNOWN_SOURCES:
                print(f"  [Warning] '{basename}' is not in the known HKEX OMDD "
                      f"source list. Indexing anyway.")

            page_docs = md_doc_conversion(file_path)
            if not page_docs:
                print(f"  Skipping – no content extracted from {file_path}")
                continue

            print(f"  Loaded {len(page_docs)} pages.")

            split_docs = split_documents(
                docs=page_docs,
                file_path=file_path,
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"],
            )
            print(f"  Generated {len(split_docs)} chunks.")
            combined_split_docs.extend(split_docs)

        if combined_split_docs:
            print(f"\n-> Indexing {len(combined_split_docs)} total chunks into Chroma...")
            vector_store.add_documents(documents=combined_split_docs)

            write_bm25_cache(
                docs=combined_split_docs,
                bm25_cache_path=config["bm25_cache_path"],
                append=not do_reset,
            )
            print("Success: indexing complete.\n")
        else:
            print("No chunks extracted. Check file contents.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting.")