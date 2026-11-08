import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.search.vector_store import vector_store

tenant_id = "558c93f9-1135-40ba-86f7-cd6676525ef1"

def search(q, doc_id, n=1):
    res = vector_store.search_hybrid(tenant_id=tenant_id, query=q, limit=n, document_id=doc_id)
    print(f"=== {q} ===")
    for r in res:
        p = r.get("page_number")
        t = r.get("text", "").replace("\n", " ")[:200]
        print(f"P{p}: {t}...\n")

if __name__ == "__main__":
    search("GDP per capita PPP Afghanistan Albania", "2ed7b57e-e464-43cc-9071-6329a7458d96")
    search("Purchasing power parity PPP definition", "2ed7b57e-e464-43cc-9071-6329a7458d96")
    search("telephone mobile contact number", "e773b7bb-a01c-49e2-b784-98aaa44f8b59")
    search("action taken police social care referral", "e773b7bb-a01c-49e2-b784-98aaa44f8b59")
    search("shared_buffers default value 128MB", "16d17c45-e073-4e5b-a8ae-65783e92a0d1")
    search("Table 27.44 CREATE INDEX Phases", "16d17c45-e073-4e5b-a8ae-65783e92a0d1")
    search("Technology Readiness Level TRL Phase B C", "1977db67-7f70-445b-b424-a4c1dbd237b7")
    search("Figure 2.2-1 Systems Engineering engine", "1977db67-7f70-445b-b424-a4c1dbd237b7")
    search("Table 1.1 evaluation criteria DAC OECD", "43efd42f-f3b0-47a5-b9dd-085ef1e41000")
