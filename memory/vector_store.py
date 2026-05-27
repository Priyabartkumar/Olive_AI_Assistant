import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, CHROMA_COLLECTION, CHROMA_PERSIST_DIR


class VectorStore:
    def __init__(self, collection_name: str = CHROMA_COLLECTION):
        self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, doc_id: str, text: str, metadata: dict | None = None):
        embedding = self._encoder.encode(text).tolist()
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        embedding = self._encoder.encode(text).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits

    def clear(self):
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()
