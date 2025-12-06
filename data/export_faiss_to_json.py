#!/usr/bin/env python3
"""
export_faiss_to_json.py

Простой скрипт для экспорта всех чанков из FAISS индекса в JSON.

Использование:
    python export_faiss_to_json.py --index_dir ./faiss_kitesurf_wiki --output chunks.json
    python export_faiss_to_json.py --index_dir ./faiss_kitesurf_wiki  # выведет в stdout
"""

import argparse
import json
import sys
from pathlib import Path


def load_faiss_index(index_dir: str):
    """Загрузить FAISS индекс."""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}", file=sys.stderr)
        print("Install with: pip install langchain-community langchain-huggingface", file=sys.stderr)
        sys.exit(1)
    
    if not Path(index_dir).exists():
        print(f"Error: Index directory not found: {index_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Инициализация embeddings (нужны для загрузки)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    # Загрузка индекса
    vectorstore = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    return vectorstore


def extract_all_chunks(vectorstore) -> list:
    """Извлечь все чанки из FAISS индекса."""
    chunks = []
    
    # Проверяем наличие index_to_docstore_id для правильного порядка
    if hasattr(vectorstore, "index_to_docstore_id"):
        # Используем порядок из FAISS индекса
        index_to_docstore_id = vectorstore.index_to_docstore_id
        docstore = vectorstore.docstore
        
        for idx in range(len(index_to_docstore_id)):
            doc_id = index_to_docstore_id[idx]
            doc = docstore.search(doc_id)
            
            chunk_data = {
                "index": idx,
                "id": doc_id,
                "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                "metadata": doc.metadata if hasattr(doc, "metadata") else {},
            }
            chunks.append(chunk_data)
    elif hasattr(vectorstore, "docstore"):
        # Fallback: извлекаем напрямую из docstore
        docstore = vectorstore.docstore
        if hasattr(docstore, "_dict"):
            for doc_id, doc in docstore._dict.items():
                chunk_data = {
                    "id": doc_id,
                    "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {},
                }
                chunks.append(chunk_data)
    
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Export all chunks from FAISS index to JSON"
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory with FAISS index",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file (if not specified, prints to stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON with indentation",
    )
    
    args = parser.parse_args()
    
    # Загрузка индекса
    print(f"Loading FAISS index from: {args.index_dir}", file=sys.stderr)
    vectorstore = load_faiss_index(args.index_dir)
    
    # Извлечение всех чанков
    print("Extracting chunks...", file=sys.stderr)
    chunks = extract_all_chunks(vectorstore)
    print(f"Extracted {len(chunks)} chunks", file=sys.stderr)
    
    # Подготовка JSON
    indent = 2 if args.pretty else None
    json_data = json.dumps(chunks, ensure_ascii=False, indent=indent)
    
    # Вывод
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_data)
        print(f"✓ Saved to: {args.output}", file=sys.stderr)
    else:
        print(json_data)
    
    print(f"✓ Done! Exported {len(chunks)} chunks", file=sys.stderr)


if __name__ == "__main__":
    main()

