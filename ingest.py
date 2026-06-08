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
_HR_RE = re.compile(r"^---+\s*$", re.MULTILINE)
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

def clean_document(text: str) -> str:
    cleaned = _HTML_COMMENT_RE.sub("", text)
    if "<!--" in cleaned:
        raise ValueError("Unclosed HTML comment in document — content would be silently lost.")
    text = cleaned
    text = _HR_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_document(text: str, source_file: str) -> list[DocumentChunk]:
    stem = Path(source_file).stem
    lines = text.splitlines()
    chunks: list[DocumentChunk] = []
    current_header = ""
    current_lines: list[str] = []
    chunk_index = 0

    def _flush(header: str, body_lines: list[str]) -> DocumentChunk | None:
        nonlocal chunk_index
        body = "\n".join(body_lines).strip()
        full_text = f"{header}\n{body}".strip() if header else body
        if len(full_text) < MIN_CHUNK_CHARS:
            return None
        chunk = DocumentChunk(
            chunk_id=f"{stem}_{chunk_index}",
            source_file=source_file,
            header=header,
            text=full_text,
            char_count=len(full_text),
        )
        chunk_index += 1
        return chunk

    for line in lines:
        if _HEADER_RE.match(line):
            chunk = _flush(current_header, current_lines)
            if chunk:
                chunks.append(chunk)
            current_header = line
            current_lines = []
        else:
            current_lines.append(line)

    chunk = _flush(current_header, current_lines)
    if chunk:
        chunks.append(chunk)

    return chunks

def load_all_documents(docs_dir: Path = DOCS_DIR) -> list[DocumentChunk]: ...

def validate_chunks(chunks: list[DocumentChunk]) -> None: ...

def load_into_chromadb(chunks: list[DocumentChunk], collection: chromadb.Collection) -> None: ...

def main() -> None: ...


if __name__ == "__main__":
    main()
