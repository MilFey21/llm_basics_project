#!/usr/bin/env python3
"""
inspect_faiss_index.py

Инспекция FAISS индекса - показывает что хранится в индексе.

Использование:
    python inspect_faiss_index.py --index_dir ./faiss_kitesurf_wiki
    python inspect_faiss_index.py --index_dir ./faiss_kitesurf_wiki --show_chunks 10
    python inspect_faiss_index.py --index_dir ./faiss_kitesurf_wiki --search "wind conditions"
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    missing = []
    
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        missing.append("langchain-community")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        missing.append("langchain-huggingface")
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Please install them with: pip install -r requirements.txt")
        return False
    
    return True


def load_faiss_index(index_dir: str):
    """Load FAISS index from disk."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    if not Path(index_dir).exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    
    logger.info(f"Loading FAISS index from: {index_dir}")
    
    # Initialize embeddings (same as used for building)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    vectorstore = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    return vectorstore


def get_index_stats(vectorstore) -> Dict[str, Any]:
    """Get statistics about the index."""
    # Access internal FAISS index
    index = vectorstore.index
    docstore = vectorstore.docstore
    
    stats = {
        "total_vectors": index.ntotal,
        "dimension": index.d,
        "total_documents": len(docstore._dict) if hasattr(docstore, "_dict") else 0,
    }
    
    return stats


def get_all_documents(vectorstore) -> List[Any]:
    """Extract all documents from the vectorstore."""
    documents = []
    
    # Access docstore
    if hasattr(vectorstore, "docstore"):
        docstore = vectorstore.docstore
        if hasattr(docstore, "_dict"):
            for doc_id, doc in docstore._dict.items():
                documents.append(doc)
    
    return documents


def analyze_metadata(documents: List[Any]) -> Dict[str, Any]:
    """Analyze metadata from all documents."""
    metadata_analysis = {
        "total_chunks": len(documents),
        "sources": Counter(),
        "parent_titles": Counter(),
        "languages": Counter(),
        "seed_queries": Counter(),
        "chunks_per_document": defaultdict(int),
    }
    
    for doc in documents:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        
        source = metadata.get("source", "unknown")
        metadata_analysis["sources"][source] += 1
        
        parent_title = metadata.get("parent_title", metadata.get("title", "unknown"))
        metadata_analysis["parent_titles"][parent_title] += 1
        
        lang = metadata.get("lang", "unknown")
        metadata_analysis["languages"][lang] += 1
        
        seed = metadata.get("seed_query", "unknown")
        metadata_analysis["seed_queries"][seed] += 1
        
        # Count chunks per document
        metadata_analysis["chunks_per_document"][parent_title] += 1
    
    return metadata_analysis


def print_header(title: str, char: str = "="):
    """Print formatted header."""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}")


def print_subheader(title: str):
    """Print formatted subheader."""
    print(f"\n{title}")
    print("-" * 60)


def display_stats(stats: Dict[str, Any]):
    """Display index statistics."""
    print_header("📊 INDEX STATISTICS")
    print(f"  Total vectors in FAISS:     {stats['total_vectors']:,}")
    print(f"  Vector dimension:           {stats['dimension']}")
    print(f"  Total documents in store:   {stats['total_documents']:,}")


def display_metadata_analysis(analysis: Dict[str, Any]):
    """Display metadata analysis."""
    print_header("📋 METADATA ANALYSIS")
    
    print(f"\n  Total chunks:               {analysis['total_chunks']:,}")
    
    # Languages
    print_subheader("🌍 Languages:")
    for lang, count in analysis["languages"].most_common():
        print(f"    • {lang}: {count} chunks")
    
    # Seed queries
    print_subheader("🔍 Seed Queries:")
    for seed, count in analysis["seed_queries"].most_common():
        print(f"    • {seed}: {count} chunks")
    
    # Parent documents
    print_subheader("📄 Documents (by chunks):")
    for title, count in analysis["parent_titles"].most_common():
        print(f"    • {title}: {count} chunks")
    
    # Sources (top 10)
    print_subheader("🔗 Top Sources:")
    for i, (source, count) in enumerate(analysis["sources"].most_common(10), 1):
        source_short = source[:60] + "..." if len(source) > 60 else source
        print(f"    [{i}] {source_short}")
        print(f"        ({count} chunks)")


def display_sample_chunks(documents: List[Any], n: int = 5):
    """Display sample chunks from the index."""
    print_header("📝 SAMPLE CHUNKS")
    
    if not documents:
        print("  No documents found.")
        return
    
    # Show first N chunks
    for i, doc in enumerate(documents[:n], 1):
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        
        print(f"\n  [{i}] Chunk #{metadata.get('chunk_id', '?')} from: {metadata.get('parent_title', 'Unknown')}")
        print(f"      Source: {metadata.get('source', 'N/A')}")
        print(f"      Seed query: {metadata.get('seed_query', 'N/A')}")
        print(f"      Content length: {len(content)} characters")
        
        # Show first 200 chars
        preview = content[:200].replace("\n", " ").strip()
        if len(content) > 200:
            preview += "..."
        print(f"      Preview: {preview}")


def display_full_chunks(documents: List[Any], n: int = 3):
    """Display full content of N chunks."""
    print_header("📖 FULL CHUNK CONTENTS", char="-")
    
    if not documents:
        print("  No documents found.")
        return
    
    for i, doc in enumerate(documents[:n], 1):
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        
        print(f"\n{'=' * 80}")
        print(f"CHUNK {i}/{min(n, len(documents))}")
        print(f"{'=' * 80}")
        print(f"📄 Document:    {metadata.get('parent_title', 'Unknown')}")
        print(f"🔢 Chunk ID:    {metadata.get('chunk_id', '?')} / {metadata.get('total_chunks', '?')}")
        print(f"🔗 Source:      {metadata.get('source', 'N/A')}")
        print(f"🔍 Seed Query:  {metadata.get('seed_query', 'N/A')}")
        print(f"🌍 Language:    {metadata.get('lang', 'N/A')}")
        print(f"📏 Length:      {len(content)} characters")
        print(f"\n{'─' * 80}")
        print("CONTENT:")
        print(f"{'─' * 80}")
        print(content)
        print(f"{'─' * 80}")


def export_to_json(documents: List[Any], output_file: str):
    """Export all chunks to JSON file."""
    export_data = []
    
    for doc in documents:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        
        export_data.append({
            "content": content,
            "metadata": metadata,
            "content_length": len(content),
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Exported {len(export_data)} chunks to: {output_file}")


def search_in_index(vectorstore, query: str, k: int = 5):
    """Perform similarity search."""
    print_header(f"🔍 SEARCH RESULTS: \"{query}\"")
    
    try:
        results = vectorstore.similarity_search(query, k=k)
        
        if not results:
            print("  No results found.")
            return
        
        for i, doc in enumerate(results, 1):
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            
            print(f"\n  [{i}] 📄 {metadata.get('parent_title', 'Unknown')} (chunk {metadata.get('chunk_id', '?')})")
            print(f"      🔗 {metadata.get('source', 'N/A')}")
            print(f"      📏 {len(content)} chars")
            
            preview = content[:300].replace("\n", " ").strip()
            if len(content) > 300:
                preview += "..."
            print(f"      📝 {preview}")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"  ✗ Search error: {e}")


def search_with_scores(vectorstore, query: str, k: int = 5):
    """Perform similarity search with scores."""
    print_header(f"🔍 SEARCH WITH SCORES: \"{query}\"")
    
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
        
        if not results:
            print("  No results found.")
            return
        
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            
            print(f"\n  [{i}] Score: {score:.4f}")
            print(f"      📄 {metadata.get('parent_title', 'Unknown')} (chunk {metadata.get('chunk_id', '?')})")
            print(f"      🔗 {metadata.get('source', 'N/A')}")
            
            preview = content[:250].replace("\n", " ").strip()
            if len(content) > 250:
                preview += "..."
            print(f"      📝 {preview}")
    except Exception as e:
        logger.error(f"Search with scores failed: {e}")
        logger.info("Falling back to regular search...")
        search_in_index(vectorstore, query, k=k)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect FAISS index and show stored data"
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory with FAISS index",
    )
    parser.add_argument(
        "--show_chunks",
        type=int,
        default=5,
        help="Number of sample chunks to display (default: 5)",
    )
    parser.add_argument(
        "--show_full",
        type=int,
        default=0,
        help="Number of chunks to display with full content (default: 0)",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Perform similarity search with this query",
    )
    parser.add_argument(
        "--search_k",
        type=int,
        default=5,
        help="Number of search results (default: 5)",
    )
    parser.add_argument(
        "--with_scores",
        action="store_true",
        help="Show similarity scores in search results",
    )
    parser.add_argument(
        "--export_json",
        type=str,
        help="Export all chunks to JSON file",
    )
    parser.add_argument(
        "--list_all",
        action="store_true",
        help="List all documents (no limit)",
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "=" * 80)
    print("🔍 FAISS INDEX INSPECTOR")
    print("=" * 80)
    print(f"  Index directory: {args.index_dir}")
    print("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Load index
    try:
        vectorstore = load_faiss_index(args.index_dir)
    except FileNotFoundError as e:
        logger.error(f"Index not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        sys.exit(1)
    
    # Get all documents
    documents = get_all_documents(vectorstore)
    logger.info(f"Extracted {len(documents)} documents from index")
    
    # Display statistics
    stats = get_index_stats(vectorstore)
    display_stats(stats)
    
    # Analyze metadata
    analysis = analyze_metadata(documents)
    display_metadata_analysis(analysis)
    
    # Show sample chunks
    if args.show_chunks > 0:
        display_sample_chunks(documents, n=args.show_chunks)
    
    # Show full chunks
    if args.show_full > 0:
        display_full_chunks(documents, n=args.show_full)
    
    # List all documents
    if args.list_all:
        print_header("📚 ALL DOCUMENTS")
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            print(f"\n  [{i}/{len(documents)}] {metadata.get('parent_title', 'Unknown')} "
                  f"(chunk {metadata.get('chunk_id', '?')})")
            print(f"      Length: {len(content)} chars | Source: {metadata.get('source', 'N/A')}")
    
    # Perform search
    if args.search:
        if args.with_scores:
            search_with_scores(vectorstore, args.search, k=args.search_k)
        else:
            search_in_index(vectorstore, args.search, k=args.search_k)
    
    # Export to JSON
    if args.export_json:
        export_to_json(documents, args.export_json)
    
    print("\n" + "=" * 80)
    logger.info("✅ Inspection complete!")


if __name__ == "__main__":
    main()

