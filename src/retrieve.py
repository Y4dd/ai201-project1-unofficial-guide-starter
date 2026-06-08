import chromadb
from src.load import CHROMA_DB_PATH, COLLECTION_NAME, make_embedding_function


def retrieve(
    query: str,
    k: int = 5,
    collection: chromadb.Collection | None = None,
) -> list[dict]:
    if collection is None:
        ef = make_embedding_function()
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    n = min(k, collection.count())
    if n == 0:
        return []
    results = collection.query(query_texts=[query], n_results=n)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    return [
        {
            "text": doc,
            "source_file": meta["source_file"],
            "header": meta["header"],
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]
