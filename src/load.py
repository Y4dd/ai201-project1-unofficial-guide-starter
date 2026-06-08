from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.chunk import DocumentChunk, clean_document, chunk_document

DOCS_DIR = Path(__file__).parent.parent / "documents"
CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "wmu_housing"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"


def make_embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def load_document(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_all_documents(docs_dir: Path = DOCS_DIR) -> list[DocumentChunk]:
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    all_chunks: list[DocumentChunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = load_document(path)
        try:
            cleaned = clean_document(raw)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        chunks = chunk_document(cleaned, path.name)
        all_chunks.extend(chunks)
    return all_chunks


def load_into_chromadb(chunks: list[DocumentChunk], collection: chromadb.Collection) -> None:
    if not chunks:
        return
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"source_file": c.source_file, "header": c.header, "char_count": c.char_count}
            for c in chunks
        ],
    )
