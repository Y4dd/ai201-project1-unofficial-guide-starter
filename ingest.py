import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DOCS_DIR = Path(__file__).parent / "documents"
CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "wmu_housing"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"

_HEADER_RE = re.compile(r"^#{2,3} ")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HR_RE = re.compile(r"^---+$", re.MULTILINE)
MIN_CHUNK_CHARS = 80


@dataclass
class DocumentChunk:
    chunk_id: str       # e.g. "source7_michigan_tenant_law_2"
    source_file: str    # e.g. "source7_michigan_tenant_law.md"
    header: str         # e.g. "## The 30-Day Return Deadline"
    text: str           # full chunk text including the header line
    char_count: int


def load_document(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def clean_document(text: str) -> str: ...

def chunk_document(text: str, source_file: str) -> list[DocumentChunk]: ...

def load_all_documents(docs_dir: Path = DOCS_DIR) -> list[DocumentChunk]: ...

def validate_chunks(chunks: list[DocumentChunk]) -> None: ...

def load_into_chromadb(chunks: list[DocumentChunk], collection: chromadb.Collection) -> None: ...

def main() -> None: ...


if __name__ == "__main__":
    main()
