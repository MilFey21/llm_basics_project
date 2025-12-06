#!/usr/bin/env python3
"""
build_faiss_wiki.py

Сборка мини-RAG индекса на FAISS из Wikipedia по теме kitesurfing.
Использует LangChain WikipediaRetriever для получения документов.

Использование:
    python build_faiss_wiki.py --lang en --index_dir ./faiss_kitesurf_wiki
    python build_faiss_wiki.py --lang ru --index_dir ./faiss_kitesurf_wiki_ru
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

# Seed queries for Wikipedia retrieval - PUBLIC INDEX
PUBLIC_QUERY_SEEDS_RU = [
  "Кайтбординг",
  "Международная ассоциация кайтсёрфинга",
  "Вейкбординг",
  "Виндсёрфинг",
  "Водные виды спорта",
  "Парусный спорт",
  "Экстремальные виды спорта",
  "Океанические течения",
  "Пассат",
  "Муссоны",
  "Прилив",
  "Отлив",
  "Тропический циклон",
  "Шквал",
  "Гидрометеорология",
  "Пляж",
  "Береговая линия",
  "Побережье",
  "Красное море",
  "Средиземное море",
  "Атлантический океан — климат",
  "Индийский океан — климат",
  "Метеорология",
  "Серфинг",
  "Парасейлинг",
  "Парапланеризм",
  "Первая помощь"
]

# Seed queries for Wikipedia retrieval - PRIVATE INDEX
PRIVATE_QUERY_SEEDS_RU = [
  "Спортивная подготовка",
  "Физическая подготовка",
  "Спортивная медицина",
  "Спортивная травма",
  "Травматология",
  "Оказание первой помощи",
  "Сердечно-лёгочная реанимация",
  "Гипотермия",
  "Обезвоживание",
  "Тепловой удар",
  "Утопление",
  "Выживание",
  "Безопасность жизнедеятельности",
  "Риск-менеджмент",
  "Шторм",
  "Цунами",
  "Морское право",
  "Спасательные работы",
  "Работа спасателя",
  "Спортивная психология",
  "Мотивация",
  "Биомеханика",
  "Координация движений",
]

# Default configuration
DEFAULT_LANG = "ru"
DEFAULT_TOP_K_PAGES = 1
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_INDEX_DIR = "./faiss_kitesurf_wiki"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Demo search queries for PUBLIC index
DEMO_QUERIES_PUBLIC = [
    "оборудование для кайтсерфинга",
    "безопасность в кайтсерфинге",
    "лучшие места для кайтсерфинга",
]

# Demo search queries for PRIVATE index
DEMO_QUERIES_PRIVATE = [
    "методика обучения кайтсерфингу",
    "профилактика травм у кайтсерферов",
    "организация спортивных мероприятий",
]


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    missing = []
    
    try:
        import langchain
    except ImportError:
        missing.append("langchain")
    
    try:
        from langchain_community.retrievers import WikipediaRetriever
    except ImportError:
        missing.append("langchain-community")
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        missing.append("langchain-text-splitters")
    
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    try:
        import wikipedia
    except ImportError:
        missing.append("wikipedia")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        missing.append("langchain-huggingface")
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Please install them with: pip install -r requirements.txt")
        return False
    
    return True


def get_embeddings(use_openai: bool = False):
    """
    Get embeddings model.
    
    If OPENAI_API_KEY is set and use_openai=True, use OpenAI embeddings.
    Otherwise, use HuggingFace embeddings (free).
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if use_openai and openai_key:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings()
        except ImportError:
            logger.warning("langchain-openai not installed, falling back to HuggingFace")
    
    from langchain_huggingface import HuggingFaceEmbeddings
    
    logger.info(f"Using HuggingFace embeddings: {DEFAULT_EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def fetch_wikipedia_documents(
    seeds: list[str],
    lang: str = "en",
    top_k: int = 1,
) -> list:
    """
    Fetch documents from Wikipedia for each seed query.
    
    Returns deduplicated list of Document objects.
    """
    from langchain_community.retrievers import WikipediaRetriever
    
    retriever = WikipediaRetriever(
        lang=lang,
        top_k_results=top_k,
        load_all_available_meta=True,  # Не загружаем лишние метаданные
        doc_content_chars_max=1000000
    )
    
    all_docs = []
    seen_hashes = set()
    
    for seed in seeds:
        logger.info(f"Fetching Wikipedia pages for: '{seed}'")
        try:
            docs = retriever.invoke(seed)
            
            for doc in docs:
                # Create hash for deduplication
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                
                if content_hash in seen_hashes:
                    logger.debug(f"Skipping duplicate: {doc.metadata.get('title', 'unknown')}")
                    continue
                
                seen_hashes.add(content_hash)
                
                # Сохраняем только основные метаданные
                original_title = doc.metadata.get("title", "unknown")
                original_summary = doc.metadata.get("summary", "")
                
                # Очищаем метаданные и оставляем только нужное
               
                
                all_docs.append(doc)
                logger.info(f"  ✓ Added: {doc.metadata.get('title', 'unknown')}")
                
        except Exception as e:
            logger.warning(f"  ✗ Failed to fetch for '{seed}': {e}")
            continue
    
    logger.info(f"Total unique documents fetched: {len(all_docs)}")
    return all_docs


def chunk_documents(
    documents: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list:
    """
    Split documents into chunks for indexing.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    if not documents:
        logger.warning("No documents to chunk!")
        return []
    
    # Filter out empty documents
    valid_docs = [doc for doc in documents if doc.page_content.strip()]
    logger.info(f"Valid documents for chunking: {len(valid_docs)}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = []
    
    for doc in valid_docs:
        doc_chunks = splitter.split_documents([doc])
        
        for i, chunk in enumerate(doc_chunks):
            # Add chunk-specific metadata
            chunk.metadata["chunk_id"] = i
            chunk.metadata["parent_title"] = doc.metadata.get("title", "unknown")
            chunk.metadata["total_chunks"] = len(doc_chunks)
            chunks.append(chunk)
    
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


def build_faiss_index(chunks: list, embeddings, index_dir: str) -> None:
    """
    Build FAISS index from chunks and save to disk.
    """
    from langchain_community.vectorstores import FAISS
    
    if not chunks:
        raise ValueError("No chunks to index!")
    
    logger.info(f"Building FAISS index with {len(chunks)} chunks...")
    
    # Create FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Ensure directory exists
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    
    # Save index
    vectorstore.save_local(index_dir)
    
    logger.info(f"✓ FAISS index saved to: {index_dir}")
    
    # Log index stats
    index_files = list(Path(index_dir).glob("*"))
    total_size = sum(f.stat().st_size for f in index_files if f.is_file())
    logger.info(f"  Index size: {total_size / 1024:.2f} KB")
    logger.info(f"  Files: {[f.name for f in index_files]}")


def load_faiss_index(index_dir: str, embeddings):
    """
    Load FAISS index from disk.
    """
    from langchain_community.vectorstores import FAISS
    
    if not Path(index_dir).exists():
        raise FileNotFoundError(f"Index directory not found: {index_dir}")
    
    logger.info(f"Loading FAISS index from: {index_dir}")
    vectorstore = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    
    return vectorstore


def demo_search(vectorstore, queries: list[str], k: int = 5) -> None:
    """
    Perform demo similarity search and display results.
    """
    print("\n" + "=" * 80)
    print("DEMO SEARCH RESULTS")
    print("=" * 80)
    
    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 60)
        
        try:
            results = vectorstore.similarity_search(query, k=k)
            
            if not results:
                print("  No results found.")
                continue
            
            for i, doc in enumerate(results, 1):
                title = doc.metadata.get("parent_title", doc.metadata.get("title", "Unknown"))
                source = doc.metadata.get("source", "N/A")
                chunk_id = doc.metadata.get("chunk_id", "?")
                
                # Get first 250 characters of content
                content_preview = doc.page_content[:250].replace("\n", " ").strip()
                if len(doc.page_content) > 250:
                    content_preview += "..."
                
                print(f"\n  [{i}] 📄 {title} (chunk {chunk_id})")
                print(f"      🔗 {source}")
                print(f"      📝 {content_preview}")
                
        except Exception as e:
            print(f"  ✗ Search failed: {e}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build PUBLIC and PRIVATE FAISS indexes from Wikipedia articles about kitesurfing"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=DEFAULT_LANG,
        choices=["ru"],
        help=f"Wikipedia language (default: {DEFAULT_LANG})",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help=f"Base directory to save FAISS indexes (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K_PAGES,
        help=f"Number of pages per seed query (default: {DEFAULT_TOP_K_PAGES})",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--use_openai",
        action="store_true",
        help="Use OpenAI embeddings if OPENAI_API_KEY is set",
    )
    parser.add_argument(
        "--skip_build",
        action="store_true",
        help="Skip building indexes, only run demo search (indexes must exist)",
    )
    parser.add_argument(
        "--demo_k",
        type=int,
        default=5,
        help="Number of results for demo search (default: 5)",
    )
    parser.add_argument(
        "--index_type",
        type=str,
        choices=["public", "private", "both"],
        default="both",
        help="Which index to build/search: public, private, or both (default: both)",
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "=" * 80)
    print("🪁 WIKIPEDIA KITESURFING RAG INDEX BUILDER (PUBLIC & PRIVATE)")
    print("=" * 80)
    print(f"  Language:      {args.lang}")
    print(f"  Base dir:      {args.index_dir}")
    print(f"  Index type:    {args.index_type}")
    print(f"  Top K pages:   {args.top_k}")
    print(f"  Chunk size:    {args.chunk_size}")
    print(f"  Chunk overlap: {args.chunk_overlap}")
    print("=" * 80 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get embeddings
    try:
        embeddings = get_embeddings(use_openai=args.use_openai)
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        sys.exit(1)
    
    # Determine which indexes to process
    indexes_to_process = []
    if args.index_type in ["public", "both"]:
        indexes_to_process.append(("public", PUBLIC_QUERY_SEEDS_RU, DEMO_QUERIES_PUBLIC))
    if args.index_type in ["private", "both"]:
        indexes_to_process.append(("private", PRIVATE_QUERY_SEEDS_RU, DEMO_QUERIES_PRIVATE))
    
    # Build or load indexes
    if not args.skip_build:
        for index_name, seeds, _ in indexes_to_process:
            print("\n" + "🔷" * 40)
            print(f"BUILDING {index_name.upper()} INDEX")
            print("🔷" * 40 + "\n")
            
            index_path = os.path.join(args.index_dir, index_name)
            
            # Step 1: Fetch documents
            logger.info("=" * 40)
            logger.info(f"STEP 1: Fetching Wikipedia documents for {index_name} index")
            logger.info("=" * 40)
            
            try:
                documents = fetch_wikipedia_documents(
                    seeds=seeds,
                    lang=args.lang,
                    top_k=args.top_k,
                )
            except Exception as e:
                logger.error(f"Failed to fetch Wikipedia documents for {index_name}: {e}")
                logger.error("Check your internet connection and try again.")
                continue
            
            if not documents:
                logger.warning(f"No documents fetched for {index_name} index. Skipping.")
                continue
            
            # Step 2: Chunk documents
            logger.info("\n" + "=" * 40)
            logger.info(f"STEP 2: Chunking documents for {index_name} index")
            logger.info("=" * 40)
            
            chunks = chunk_documents(
                documents=documents,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            
            if not chunks:
                logger.warning(f"No chunks created for {index_name} index. Skipping.")
                continue
            
            # Step 3: Build and save FAISS index
            logger.info("\n" + "=" * 40)
            logger.info(f"STEP 3: Building FAISS index for {index_name}")
            logger.info("=" * 40)

            try:
                build_faiss_index(chunks, embeddings, index_path)
            except Exception as e:
                logger.error(f"Failed to build FAISS index for {index_name}: {e}")
                continue
    
    # Step 4: Demo search
    print("\n" + "🔷" * 40)
    print("DEMO SEARCH")
    print("🔷" * 40 + "\n")
    
    for index_name, _, demo_queries in indexes_to_process:
        index_path = os.path.join(args.index_dir, index_name)
        
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Testing {index_name.upper()} index")
        logger.info(f"{'=' * 40}")
        
        try:
            vectorstore = load_faiss_index(index_path, embeddings)
            demo_search(vectorstore, demo_queries, k=args.demo_k)
            
        except FileNotFoundError as e:
            logger.error(f"{index_name.upper()} index not found: {e}")
            logger.error("Run without --skip_build to create the index first.")
            continue
        except Exception as e:
            logger.error(f"Demo search failed for {index_name}: {e}")
            continue
    
    logger.info("\n✅ All done!")


if __name__ == "__main__":
    main()

