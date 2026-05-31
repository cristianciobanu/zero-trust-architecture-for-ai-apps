"""
Document ingestion script.

Reads PDF files, splits them into chunks, generates embeddings,
and uploads them to the Qdrant vector database.

Usage:
    python ingest.py /path/to/documents/
    python ingest.py /path/to/single_file.pdf
"""

import logging
import os
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant.qdrant.svc.cluster.local")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "financial_docs")
EMBED_MODEL = "/app/models/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_pdfs(path: Path) -> list[dict]:
    """Load all PDFs from a file or directory and return a list of page texts."""
    pdf_files = [path] if path.is_file() else sorted(path.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", path)
        return []

    documents = []
    for pdf_file in pdf_files:
        logger.info("Loading %s", pdf_file.name)
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()
        for page in pages:
            documents.append({
                "text": page.page_content,
                "source": f"{pdf_file.name}, page {page.metadata.get('page', '?') + 1}",
            })

    logger.info("Loaded %d pages from %d PDF(s)", len(documents), len(pdf_files))
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for split in splits:
            if split.strip():
                chunks.append({"text": split.strip(), "source": doc["source"]})

    logger.info("Created %d chunks (chunk_size=%d, overlap=%d)", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def embed_and_upload(chunks: list[dict]) -> None:
    """Generate embeddings and upload to Qdrant."""
    logger.info("Loading embedding model: %s", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    logger.info("Generating embeddings for %d chunks...", len(chunks))
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)

    logger.info("Uploading to Qdrant (%s:%d, collection=%s)", QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    points = [
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={"text": chunks[i]["text"], "source": chunks[i]["source"]},
        )
        for i, embedding in enumerate(embeddings)
    ]

    # Upload in batches of 100
    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info("Uploaded batch %d-%d", start, start + len(batch))

    logger.info("Ingestion complete: %d chunks uploaded", len(points))


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path-to-pdf-or-directory>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Path not found: {path}")
        sys.exit(1)

    documents = load_pdfs(path)
    if not documents:
        sys.exit(1)

    chunks = chunk_documents(documents)
    embed_and_upload(chunks)


if __name__ == "__main__":
    main()
