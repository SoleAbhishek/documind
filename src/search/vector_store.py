import os
import uuid
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client import QdrantClient, models

from src.config import settings
from src.chunking.chunker import Chunk

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class DenseEmbeddingWrapper:
    """
    Unified dense embedder wrapper supporting:
    1. Google Gemini Embeddings (gemini-embedding-001 or gemini-embedding-2) - configurable dims (default 768 or 3072)
    2. Fallback to local FastEmbed (BAAI/bge-small-en-v1.5) - 384 dims
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-embedding-001", target_dim: int = 768):
        self.api_key = api_key
        self.model_name = model_name.replace("models/", "") if model_name else "gemini-embedding-001"
        self.target_dim = target_dim
        self.is_gemini = False
        self._genai_client = None
        self._fallback_embedder = None
        self._init_embedder()

    def _init_embedder(self):
        if self.api_key and not self.api_key.startswith("AIzaSy-your") and len(self.api_key) > 20:
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.api_key)
                self.is_gemini = True
                print(f"[VectorStore] Active Dense Embedder: Google Gemini ({self.model_name}) - {self.target_dim} dims")
                return
            except Exception as e:
                print(f"[VectorStore] Failed to initialize Gemini embedder: {e}. Falling back to local BAAI/bge-small-en-v1.5...")

        from fastembed import TextEmbedding
        print("[VectorStore] Active Dense Embedder: Local FastEmbed (BAAI/bge-small-en-v1.5) - 384 dims")
        self._fallback_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.is_gemini = False

    @property
    def dimension(self) -> int:
        return self.target_dim if self.is_gemini else 384

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.is_gemini:
            try:
                config = {"output_dimensionality": self.target_dim} if self.target_dim else None
                response = self._genai_client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=config
                )
                return [list(e.values) for e in response.embeddings]
            except Exception as e:
                print(f"[VectorStore] Gemini API call failed ({e}). Falling back dynamically to local embedder.")
                if self._fallback_embedder is None:
                    from fastembed import TextEmbedding
                    self._fallback_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                return [v.tolist() for v in self._fallback_embedder.embed(texts)]
        else:
            if self._fallback_embedder is None:
                from fastembed import TextEmbedding
                self._fallback_embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            return [v.tolist() for v in self._fallback_embedder.embed(texts)]


class QdrantVectorStore:
    """
    Manages Dual-Vector (Dense + Sparse) storage and Hybrid Retrieval in Qdrant.
    
    Collection Architecture:
    - Named Vector 'dense': 768 dims (Gemini) or 384 dims (local BAAI/bge-small)
    - Named Vector 'sparse': Token frequency inverted index (Qdrant/bm25)
    - Payload index: 'tenant_id' for multi-tenant data isolation
    """

    def __init__(self):
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._dense_model = None
        self._sparse_model = None
        self.dense_dim = 384  # Default, adjusted on dense_model initialization
        
        # Connect to Qdrant (Docker with in-memory fallback)
        try:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                prefer_grpc=False,
                check_compatibility=False,
                timeout=3
            )
            self.client.get_collections()
            print(f"[VectorStore] Connected to Qdrant server at {settings.QDRANT_URL}")
        except Exception as e:
            print(f"[VectorStore] Could not connect to Qdrant server ({e}). Falling back to embedded in-memory mode (:memory:).")
            self.client = QdrantClient(location=":memory:")

        # Ensure collection exists
        self._ensure_collection()

    @property
    def dense_model(self):
        """Lazy-load dense embedder on first use."""
        if self._dense_model is None:
            self._dense_model = DenseEmbeddingWrapper(
                api_key=settings.GEMINI_API_KEY,
                model_name=settings.GEMINI_EMBEDDING_MODEL,
                target_dim=settings.DENSE_VECTOR_DIM
            )
            self.dense_dim = self._dense_model.dimension
        return self._dense_model

    @property
    def sparse_model(self):
        """Lazy-load sparse BM25 embedder on first use."""
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            print("[VectorStore] Loading Sparse BM25 embedder (Qdrant/bm25)...")
            self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._sparse_model

    def _ensure_collection(self):
        """Creates the dual-vector collection if it does not already exist, or updates dimension."""
        # Touch dense_model to know active dimension
        expected_dim = self.dense_model.dimension
        self.dense_dim = expected_dim
        try:
            if self.client.collection_exists(self.collection_name):
                info = self.client.get_collection(self.collection_name)
                dense_cfg = info.config.params.vectors.get("dense")
                existing_dim = dense_cfg.size if dense_cfg else None
                if existing_dim and existing_dim != self.dense_dim:
                    print(f"[VectorStore] Collection has dense_dim={existing_dim}, expected {self.dense_dim}. Recreating collection...")
                    self.client.delete_collection(self.collection_name)

            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self.dense_dim,
                            distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    }
                )
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="tenant_id",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception:
                    pass
                print(f"[VectorStore] Initialized collection '{self.collection_name}' with dual vectors (dense_dim={self.dense_dim}).")
        except Exception as e:
            print(f"[VectorStore] Warning during collection check: {e}")

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], List[models.SparseVector]]:
        """Computes both Dense and Sparse representations for a list of strings."""
        if not texts:
            return [], []

        dense_vecs = self.dense_model.embed(texts)
        sparse_raw = list(self.sparse_model.embed(texts))
        
        sparse_vecs = [
            models.SparseVector(
                indices=item.indices.tolist(),
                values=item.values.tolist()
            )
            for item in sparse_raw
        ]
        return dense_vecs, sparse_vecs

    def upsert_chunks(self, tenant_id: str, document_id: str, chunks: List[Chunk]) -> List[str]:
        """
        Embeds and stores chunks into Qdrant. Returns list of generated qdrant_point_ids.
        """
        if not chunks:
            return []

        texts_to_embed = [c.text for c in chunks]
        dense_vecs, sparse_vecs = self.embed_texts(texts_to_embed)

        points: List[models.PointStruct] = []
        point_ids: List[str] = []

        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            point = models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vecs[i],
                    "sparse": sparse_vecs[i]
                },
                payload={
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "header_path": chunk.header_path,
                    "text": chunk.text,
                    "raw_text": chunk.raw_text,
                    "has_table": chunk.metadata.get("has_table", False)
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"[VectorStore] Upserted {len(points)} chunks to Qdrant for document {document_id}.")
        return point_ids

    def search_hybrid(
        self,
        tenant_id: str,
        query: str,
        limit: int = 5,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a native Qdrant Hybrid RRF query strictly filtered by tenant_id.
        """
        q_dense, q_sparse = self.embed_texts([query])
        
        must_conditions = [
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
        ]
        if document_id:
            must_conditions.append(
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            )

        filter_condition = models.Filter(must=must_conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=q_dense[0],
                    using="dense",
                    filter=filter_condition,
                    limit=limit * 2
                ),
                models.Prefetch(
                    query=q_sparse[0],
                    using="sparse",
                    filter=filter_condition,
                    limit=limit * 2
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit
        )

        results = []
        for p in response.points:
            results.append({
                "point_id": str(p.id),
                "rrf_score": float(p.score) if p.score is not None else 0.0,
                "document_id": p.payload.get("document_id"),
                "chunk_index": p.payload.get("chunk_index"),
                "page_number": p.payload.get("page_number"),
                "header_path": p.payload.get("header_path"),
                "text": p.payload.get("text"),
                "raw_text": p.payload.get("raw_text")
            })
        return results

    def delete_document_points(self, tenant_id: str, document_id: str):
        """Wipes all vectors belonging to a deleted document."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)),
                        models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
                    ]
                )
            )
        )
        print(f"[VectorStore] Deleted vectors for document {document_id}.")


# Global singleton instance
vector_store = QdrantVectorStore()
