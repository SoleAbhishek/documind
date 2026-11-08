import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.search.vector_store import vector_store

tenant_id = '558c93f9-1135-40ba-86f7-cd6676525ef1'

def peek_doc(doc_name, query, doc_id=None):
    res = vector_store.search_hybrid(tenant_id=tenant_id, query=query, limit=2, document_id=doc_id)
    print(f"=== {doc_name} (Query: '{query}') ===")
    for idx, r in enumerate(res, 1):
        text = r.get("text", "").replace("\n", " ")
        print(f"[{idx}] Page {r.get('page_number')}: {text[:200]}...\n")

if __name__ == "__main__":
    peek_doc("WV.1 Economy Table", "Mozambique gross domestic product", "2ed7b57e-e464-43cc-9071-6329a7458d96")
    peek_doc("Handwritten Form", "name of child date of birth", "e773b7bb-a01c-49e2-b784-98aaa44f8b59")
    peek_doc("PostgreSQL Docs", "CREATE INDEX CONCURRENTLY", "16d17c45-e073-4e5b-a8ae-65783e92a0d1")
    peek_doc("NASA Handbook", "Pre-Phase A Concept Studies", "1977db67-7f70-445b-b424-a4c1dbd237b7")
    peek_doc("Road to Results", "Evaluation capacity development diagnostic checklist", "43efd42f-f3b0-47a5-b9dd-085ef1e41000")
