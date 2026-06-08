import re
from dataclasses import dataclass
from pathlib import Path

_HEADER_RE = re.compile(r"^#{2,3} ")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HR_RE = re.compile(r"^---+\s*$", re.MULTILINE)
MIN_CHUNK_CHARS = 80


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    header: str
    text: str
    char_count: int


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


def validate_chunks(chunks: list[DocumentChunk]) -> None:
    if not chunks:
        raise ValueError("No chunks produced — check documents/ path and that *.md files exist")
    for chunk in chunks:
        if chunk.char_count <= 0:
            raise ValueError(f"Invalid char_count on {chunk.chunk_id}: {chunk.char_count}")
        if len(chunk.text) != chunk.char_count:
            raise ValueError(f"char_count mismatch on {chunk.chunk_id}")

    print(f"\n=== Validation: {len(chunks)} total chunks across all documents ===\n")
    seen_files: set[str] = set()
    samples: list[DocumentChunk] = []
    for c in chunks:
        if c.source_file not in seen_files:
            samples.append(c)
            seen_files.add(c.source_file)
        if len(samples) == 5:
            break
    for i, c in enumerate(samples, 1):
        print(f"{'─' * 60}")
        print(f"[{i}] {c.chunk_id}  |  {c.char_count} chars  |  {c.source_file}")
        print(f"Header: {c.header!r}")
        print()
        print(c.text[:400])
        if c.char_count > 400:
            print(f"  ... ({c.char_count - 400} more chars)")
        print()
    print(f"{'─' * 60}")
    print("Validation passed.\n")
