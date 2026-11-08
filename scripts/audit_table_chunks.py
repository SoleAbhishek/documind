import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from src.config import settings

def audit_table_chunks():
    print("=" * 80)
    print("FORENSIC AUDIT: HOW TABLES ARE ACTUALLY STORED IN QDRANT")
    print("=" * 80)
    
    client = QdrantClient(url=settings.QDRANT_URL, prefer_grpc=False, timeout=10)
    
    # 1. Search for Table 2.3-1 in NASA document
    print("\n--- TEST CASE 1: NASA Handbook 'Table 2.3-1' (Multi-row Lifecycle Table) ---")
    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter={
            "must": [
                {"key": "header_path", "match": {"value": "2.3 Example of Using the SE Engine > Table 2.3-1 Project Life-Cycle Phases"}}
            ]
        },
        limit=5,
        with_payload=True
    )
    
    print(f"Total chunks found under Table 2.3-1 header: {len(points)}")
    for i, p in enumerate(points):
        text = p.payload.get("text", "")
        raw = p.payload.get("raw_text", "")
        lines = text.split("\n")
        print(f"\n[Chunk {i+1}] Point ID: {p.id}")
        print(f"  Page: {p.payload.get('page_number')} | Total chars: {len(text)} | Total lines: {len(lines)}")
        print(f"  First 3 lines:")
        for line in lines[:3]:
            print(f"    {line[:120]}")
        print(f"  Does it have table header (| Phase | Purpose...)? {'| Phase |' in text}")
        print(f"  Which phases are present in this chunk?")
        phases = [ph for ph in ["Pre-Phase A", "Phase A", "Phase B", "Phase C", "Phase D", "Phase E", "Phase F"] if ph in text]
        print(f"    -> {phases}")

    # 2. Search in WV.1 Size of the economy
    print("\n\n--- TEST CASE 2: WV.1 Size of the Economy (Financial / Statistical Table) ---")
    points_econ, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter={
            "must": [
                {"key": "document_id", "match": {"value": "2ed7b57e-e464-43cc-9071-6329a7458d96"}},
                {"key": "has_table", "match": {"value": True}}
            ]
        },
        limit=3,
        with_payload=True
    )
    print(f"Total table chunks in Economic Indicators: {len(points_econ)}")
    for i, p in enumerate(points_econ):
        text = p.payload.get("text", "")
        lines = text.split("\n")
        print(f"\n[Econ Chunk {i+1}] Point ID: {p.id} | Page: {p.payload.get('page_number')}")
        print(f"  Lines count: {len(lines)} | First 4 lines:")
        for line in lines[:4]:
            print(f"    {line[:120]}")

if __name__ == "__main__":
    audit_table_chunks()
