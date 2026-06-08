import chromadb

from src.chunk import validate_chunks
from src.load import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    load_all_documents,
    load_into_chromadb,
    make_embedding_function,
)


def main() -> None:
    print("Loading and chunking documents...")
    chunks = load_all_documents()
    validate_chunks(chunks)

    print(f"Loading {len(chunks)} chunks into ChromaDB at {CHROMA_DB_PATH}...")
    ef = make_embedding_function()
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )
    load_into_chromadb(chunks, collection)
    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
