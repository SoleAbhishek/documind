"""
DocuMind Chunk Inspector Utility
Inspect chunks stored in Qdrant to verify table and image extraction fidelity.
"""
import argparse
import sys
from pathlib import Path

# Force UTF-8 on Windows terminal output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient, models
from src.config import settings


def inspect_chunks(tenant_id: str = None, filter_tables_only: bool = False, limit: int = 10):
    client = QdrantClient(url=settings.QDRANT_URL, prefer_grpc=False, timeout=5)
    
    if not client.collection_exists(settings.QDRANT_COLLECTION_NAME):
        print(f"Collection '{settings.QDRANT_COLLECTION_NAME}' does not exist.")
        return

    must_conditions = []
    if tenant_id:
        must_conditions.append(models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id)))
    if filter_tables_only:
        must_conditions.append(models.FieldCondition(key="has_table", match=models.MatchValue(value=True)))

    filter_cond = models.Filter(must=must_conditions) if must_conditions else None

    response = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=filter_cond,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )

    points, _ = response
    print("=" * 80)
    print(f"DocuMind Qdrant Chunk Inspector (Found {len(points)} matching chunks)")
    print("=" * 80)

    if not points:
        print("No matching chunks found with the specified filters.")
        return

    for idx, p in enumerate(points, 1):
        payload = p.payload or {}
        has_table = payload.get("has_table", False)
        text = payload.get("text", "")
        header_path = payload.get("header_path", "None")
        page = payload.get("page_number", "Unknown")
        doc_id = payload.get("document_id", "Unknown")
        tenant = payload.get("tenant_id", "Unknown")

        # Check for image or diagram tags/captions
        has_image = "![" in text or "[Image:" in text or "Figure" in text or "Diagram" in text

        print(f"\n[{idx}] Point ID: {p.id}")
        print(f"    Tenant:      {tenant}")
        print(f"    Document ID: {doc_id} | Page: {page}")
        print(f"    Header Path: {header_path}")
        print(f"    Has Table:   {'YES [table detected]' if has_table else 'No'}")
        print(f"    Has Image:   {'YES [image/diagram notation detected]' if has_image else 'No'}")
        print("-" * 80)
        print("    Chunk Content Preview:")
        # Indent lines for readability
        for line in text.splitlines():
            print(f"    | {line}")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect chunks in Qdrant for tables and images.")
    parser.add_argument("--tenant", type=str, default=None, help="Filter by tenant ID")
    parser.add_argument("--tables-only", action="store_true", help="Only show chunks containing tables")
    parser.add_argument("--limit", type=int, default=10, help="Number of chunks to display (default: 10)")
    args = parser.parse_args()

    inspect_chunks(tenant_id=args.tenant, filter_tables_only=args.tables_only, limit=args.limit)
