import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from httpx import AsyncClient, ASGITransport
from src.main import app
from src.db.session import init_db


@pytest.mark.asyncio
async def test_full_api_pipeline():
    print("=" * 70)
    print("DocuMind FastAPI REST API Integration Test")
    print("=" * 70)

    # Initialize DB schemas
    await init_db()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health check
        print("[1/6] Testing GET /health...")
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        print(f"Health response: {health_data}")

        # 2. Create Tenant
        print("\n[2/6] Testing POST /api/v1/tenants...")
        tenant_resp = await client.post("/api/v1/tenants", json={"name": "API Test Enterprise"})
        assert tenant_resp.status_code == 201
        tenant_data = tenant_resp.json()
        tenant_id = tenant_data["id"]
        print(f"Created Tenant: {tenant_id} ({tenant_data['name']})")

        # 3. List Tenants
        print("\n[3/6] Testing GET /api/v1/tenants...")
        list_tenants_resp = await client.get("/api/v1/tenants")
        assert list_tenants_resp.status_code == 200
        tenants = list_tenants_resp.json()
        assert any(t["id"] == tenant_id for t in tenants)
        print(f"Retrieved {len(tenants)} active tenants.")

        # 4. Upload Document
        print("\n[4/6] Testing POST /api/v1/documents/upload...")
        sample_doc = b"""# Company Security & Password Policy
## Password Complexity
Passwords must be at least 16 characters in length and contain upper, lower, numbers, and symbols.
## Multi-Factor Authentication
MFA is strictly mandatory for all employees accessing corporate systems.
"""
        files = {
            "file": ("Security_Policy.md", sample_doc, "text/markdown")
        }
        data = {
            "tenant_id": tenant_id
        }
        upload_resp = await client.post("/api/v1/documents/upload", data=data, files=files)
        if upload_resp.status_code != 202:
            print("Upload Error status:", upload_resp.status_code)
            print("Upload Error body:", upload_resp.text)
        assert upload_resp.status_code == 202
        upload_data = upload_resp.json()
        doc_id = upload_data["document_id"]
        print(f"Uploaded Document: {doc_id} (status: {upload_data['status']})")

        # 5. List Documents
        print("\n[5/6] Testing GET /api/v1/documents...")
        docs_resp = await client.get(f"/api/v1/documents?tenant_id={tenant_id}")
        assert docs_resp.status_code == 200
        docs = docs_resp.json()
        assert len(docs) >= 1
        print(f"Retrieved {len(docs)} documents for tenant {tenant_id}.")

        # 6. Test Chat Endpoint (Synchronous)
        print("\n[6/6] Testing POST /api/v1/chat...")
        chat_payload = {
            "tenant_id": tenant_id,
            "query": "What is the minimum password length requirement?",
            "top_k": 3
        }
        chat_resp = await client.post("/api/v1/chat", json=chat_payload)
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        print(f"Chat Session ID: {chat_data['session_id']}")
        print(f"Answer: {chat_data['answer']}")
        print(f"Citations count: {len(chat_data['citations'])}")

    print("\n" + "=" * 70)
    print("ALL FASTAPI REST API TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_full_api_pipeline())
