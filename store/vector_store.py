"""
store/vector_store.py
Abstraction layer over the vector database.

Phase 1  →  ChromaDB  (local, zero-infra, persistent on disk)
Phase 1+ →  Weaviate  (swap backend key in config/phase1.yaml)

The VectorStore interface is intentionally minimal so the retrieval layer
never touches database internals directly.
"""
from __future__ import annotations

from typing import List, Optional

from config.loader import VectorStoreConfig
from ingestion.chunker import Chunk


class RetrievedChunk:
    """A chunk returned by a similarity search, decorated with a score."""

    def __init__(self, chunk_dict: dict, score: float):
        self.text: str          = chunk_dict["text"]
        self.doc_id: str        = chunk_dict["doc_id"]
        self.chunk_index: int   = chunk_dict["chunk_index"]
        self.source: str        = chunk_dict["source"]
        self.citation_id: str   = chunk_dict["citation_id"]
        self.token_count: int   = chunk_dict.get("token_count", 0)
        self.score: float       = score
        self.metadata: dict     = {
            k: v for k, v in chunk_dict.items()
            if k not in {"text", "doc_id", "chunk_index", "source",
                         "citation_id", "token_count"}
        }

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(citation={self.citation_id!r}, "
            f"score={self.score:.4f}, tokens={self.token_count})"
        )


class BaseVectorStore:
    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        raise NotImplementedError

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        raise NotImplementedError

    def delete_collection(self) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class ChromaVectorStore(BaseVectorStore):
    """
    Persists chunks + embeddings in a local ChromaDB collection.
    Data survives restarts — re-ingestion only needed for new documents.
    """

    def __init__(self, cfg: VectorStoreConfig):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Run: pip install chromadb")

        chroma_cfg = cfg.chroma
        self._client = chromadb.PersistentClient(
            path=chroma_cfg.persist_directory
        )
        self._collection = self._client.get_or_create_collection(
            name=chroma_cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Upsert chunks into Chroma.
        IDs are derived from doc_id + chunk_index so re-ingestion is idempotent.
        """
        if not chunks:
            return

        ids         = [f"{c.doc_id}_{c.chunk_index}" for c in chunks]
        documents   = [c.text for c in chunks]
        metadatas   = [c.to_dict() for c in chunks]

        for m in metadatas:
            m.pop("text", None)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        """Return top_k most similar chunks above score_threshold."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - (dist / 2.0)

            if score < score_threshold:
                continue

            chunk_dict = {"text": doc, **meta}
            retrieved.append(RetrievedChunk(chunk_dict, score))

        return retrieved

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection.name)

    def count(self) -> int:
        return self._collection.count()


class WeaviateVectorStore(BaseVectorStore):
    """
    Weaviate backend.
    Requires a running Weaviate instance (Docker or Weaviate Cloud).

    Quick start:
        docker run -d -p 8080:8080 semitechnologies/weaviate:latest
    """

    def __init__(self, cfg: VectorStoreConfig):
        try:
            import weaviate
        except ImportError:
            raise ImportError("Run: pip install weaviate-client")

        wv_cfg = cfg.weaviate
        self._client = weaviate.connect_to_local(
            host=wv_cfg.url.replace("http://", "").split(":")[0],
            port=int(wv_cfg.url.split(":")[-1]),
        )
        self._class_name = wv_cfg.class_name
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create Weaviate class if it doesn't exist."""
        import weaviate.classes as wvc

        if not self._client.collections.exists(self._class_name):
            self._client.collections.create(
                name=self._class_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                properties=[
                    wvc.config.Property(name="text",        data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="doc_id",      data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
                    wvc.config.Property(name="source",      data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="citation_id", data_type=wvc.config.DataType.TEXT),
                ],
            )

    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        collection = self._client.collections.get(self._class_name)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, embeddings):
                batch.add_object(
                    properties=chunk.to_dict(),
                    vector=vector,
                )

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        collection = self._client.collections.get(self._class_name)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=["certainty"],
        )
        retrieved = []
        for obj in response.objects:
            score = obj.metadata.certainty or 0.0
            if score >= score_threshold:
                retrieved.append(RetrievedChunk(obj.properties, score))
        return retrieved

    def delete_collection(self) -> None:
        self._client.collections.delete(self._class_name)

    def count(self) -> int:
        collection = self._client.collections.get(self._class_name)
        return collection.aggregate.over_all(total_count=True).total_count


def build_vector_store(cfg: VectorStoreConfig) -> BaseVectorStore:
    if cfg.backend == "chroma":
        return ChromaVectorStore(cfg)
    elif cfg.backend == "weaviate":
        return WeaviateVectorStore(cfg)
    else:
        raise ValueError(f"Unknown vector store backend: {cfg.backend}")
