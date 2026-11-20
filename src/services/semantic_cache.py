import os
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from qdrant_client import QdrantClient, models

from src.config import settings
from src.search.vector_store import vector_store

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class SemanticCacheService:
    """
    Enterprise-grade Multi-Tenant Semantic Caching Engine.
    
    Architecture:
    1. Dedicated Qdrant collection: 'documind_semantic_cache'
    2. Vector: 768 dimensions (Gemini Embedding) or 384 dims (Local FastEmbed), Cosine distance
    3. Payload Indexing:
       - 'tenant_id': Strict isolation boundary
       - 'document_id': Document-level filter
    4. Adaptive Threshold:
       - Default threshold: 0.95 (High precision, zero hallucination transfer)
    """

    def __init__(self, collection_name: str = "documind_semantic_cache", threshold: float = 0.95):
        self.collection_name = collection_name
        self.default_threshold = threshold
        self.client = vector_store.client
        self._ensure_cache_collection()

    def _ensure_cache_collection(self):
        """Ensures the semantic cache collection exists in Qdrant with appropriate indexes."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            dim = vector_store.dense_model.dimension

            if self.collection_name not in collections:
                print(f"[SemanticCache] Initializing collection '{self.collection_name}' (dim={dim}, Cosine)...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE
                    )
                )

                # Create payload index for tenant isolation
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="tenant_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                print(f"[SemanticCache] Created collection '{self.collection_name}' successfully.")
        except Exception as e:
            print(f"[SemanticCache] Warning: Error checking/creating cache collection: {e}")

    @traceable(run_type="retriever", name="Semantic Cache Lookup")
    def lookup(
        self,
        tenant_id: str,
        query: str,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Performs semantic vector search on cached queries for a given tenant.
        Returns cached response and citations if similarity >= threshold, else None.
        """
        threshold = similarity_threshold or self.default_threshold
        query_vector = vector_store.dense_model.embed([query])[0]

        # Construct tenant isolation filter
        must_filters = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id)
            )
        ]

        if document_id:
            must_filters.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            )

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=models.Filter(must=must_filters),
                limit=1,
                with_payload=True
            )

            if not response.points:
                return None

            best_match = response.points[0]
            similarity_score = float(best_match.score)

            # Check if match meets confidence threshold
            if similarity_score >= threshold:
                payload = best_match.payload or {}
                point_id = str(best_match.id)

                # Increment hit counter asynchronously/in-place
                try:
                    hit_count = payload.get("hit_count", 0) + 1
                    self.client.set_payload(
                        collection_name=self.collection_name,
                        payload={"hit_count": hit_count, "last_accessed_at": datetime.now(timezone.utc).isoformat()},
                        points=[best_match.id]
                    )
                except Exception:
                    pass

                return {
                    "cache_hit": True,
                    "similarity": round(similarity_score, 4),
                    "original_query": payload.get("query_text", ""),
                    "answer": payload.get("answer_text", ""),
                    "citations": payload.get("citations", []),
                    "created_at": payload.get("created_at"),
                    "hit_count": payload.get("hit_count", 1),
                    "point_id": point_id
                }

            return None
        except Exception as e:
            print(f"[SemanticCache] Lookup error: {e}")
            return None

    @traceable(run_type="tool", name="Semantic Cache Store")
    def store(
        self,
        tenant_id: str,
        query: str,
        answer: str,
        citations: List[Dict[str, Any]],
        document_id: Optional[str] = None
    ) -> str:
        """
        Stores a generated query-answer pair in the semantic cache.
        """
        if not query or not answer:
            return ""

        query_vector = vector_store.dense_model.embed([query])[0]
        point_id = str(uuid.uuid4())

        payload = {
            "point_id": point_id,
            "tenant_id": tenant_id,
            "document_id": document_id,
            "query_text": query,
            "answer_text": answer,
            "citations": citations,
            "hit_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_accessed_at": datetime.now(timezone.utc).isoformat()
        }

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=query_vector,
                        payload=payload
                    )
                ]
            )
            return point_id
        except Exception as e:
            print(f"[SemanticCache] Store error: {e}")
            return ""

    def invalidate(self, tenant_id: str, document_id: Optional[str] = None) -> int:
        """
        Invalidates cached entries when documents are deleted or modified.
        """
        must_filters = [
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id)
            )
        ]
        if document_id:
            must_filters.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            )

        try:
            res = self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=must_filters)
                )
            )
            return 1
        except Exception as e:
            print(f"[SemanticCache] Invalidate error: {e}")
            return 0


# Global Semantic Cache Singleton
semantic_cache = SemanticCacheService()
