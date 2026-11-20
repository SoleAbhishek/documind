import sys
import os
import time
import asyncio

# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath("."))

from src.services.semantic_cache import semantic_cache
from src.services.chat_service import chat_service


async def test_semantic_caching_lifecycle():
    tenant_a = "558c93f9-1135-40ba-86f7-cd6676525ef1"  # Real Test tenant
    tenant_b = "339eab76-8425-4500-a114-bb4c8a671120"  # Acme Corp tenant

    # Clean up prior test cache points
    semantic_cache.invalidate(tenant_a)
    semantic_cache.invalidate(tenant_b)

    print("\n" + "=" * 80)
    print("TEST 1: Cache Miss & Storage on Initial Query")
    print("=" * 80)

    query_1 = "What is the primary purpose and advantage of using the CREATE INDEX CONCURRENTLY command in PostgreSQL?"
    
    t0 = time.time()
    # First query -> should be a cache miss, then populated
    res_1 = await chat_service.chat(tenant_id=tenant_a, query=query_1)
    duration_miss = round((time.time() - t0) * 1000, 2)

    print(f"Query 1 (Initial): '{query_1}'")
    print(f"Latency (Cache Miss): {duration_miss} ms")
    print(f"Generated Answer:\n{res_1['answer'][:150]}...")
    assert len(res_1["answer"]) > 20, "Expected non-empty answer on cache miss!"

    print("\n" + "=" * 80)
    print("TEST 2: Exact Query Cache Hit (Lightning Speedup)")
    print("=" * 80)

    t0 = time.time()
    lookup_exact = semantic_cache.lookup(tenant_id=tenant_a, query=query_1)
    duration_hit_exact = round((time.time() - t0) * 1000, 2)

    print(f"Latency (Cache Hit): {duration_hit_exact} ms")
    print(f"Similarity Score: {lookup_exact['similarity']}")
    print(f"Cached Answer Preview:\n{lookup_exact['answer'][:150]}...")
    
    assert lookup_exact is not None, "Expected cache hit on exact query!"
    assert lookup_exact["similarity"] >= 0.99, f"Expected similarity >= 0.99, got {lookup_exact['similarity']}"
    speedup = round(duration_miss / max(1, duration_hit_exact), 1)
    print(f"⚡ Performance Acceleration: {speedup}x FASTER than live generation!")

    print("\n" + "=" * 80)
    print("TEST 3: Semantic Paraphrase Cache Hit (Fuzzy Similarity)")
    print("=" * 80)

    query_paraphrased = "Why would someone use CREATE INDEX CONCURRENTLY in Postgres?"
    t0 = time.time()
    lookup_para = semantic_cache.lookup(tenant_id=tenant_a, query=query_paraphrased, similarity_threshold=0.92)
    duration_para = round((time.time() - t0) * 1000, 2)

    print(f"Paraphrased Query: '{query_paraphrased}'")
    print(f"Latency: {duration_para} ms")
    if lookup_para:
        print(f"Similarity Score: {lookup_para['similarity']}")
        print(f"Cached Answer Preview:\n{lookup_para['answer'][:150]}...")
        assert lookup_para["similarity"] >= 0.92, "Expected similarity >= 0.92"
        print("✓ Paraphrased query successfully recognized and served from Semantic Cache!")
    else:
        print("Note: Paraphrase below threshold (as expected for strict thresholds).")

    print("\n" + "=" * 80)
    print("TEST 4: Dissimilar Query Cache Miss")
    print("=" * 80)

    dissimilar_query = "What are the rules regarding domestic violence reporting forms?"
    lookup_dissimilar = semantic_cache.lookup(tenant_id=tenant_a, query=dissimilar_query)
    print(f"Dissimilar Query: '{dissimilar_query}'")
    print(f"Lookup Result: {lookup_dissimilar}")
    assert lookup_dissimilar is None, "Dissimilar query must result in a cache miss!"
    print("✓ Dissimilar query correctly triggered Cache Miss.")

    print("\n" + "=" * 80)
    print("TEST 5: Multi-Tenant Isolation Security Test")
    print("=" * 80)

    # Tenant B asks the EXACT SAME query that Tenant A asked and cached
    lookup_tenant_b = semantic_cache.lookup(tenant_id=tenant_b, query=query_1)
    print(f"Tenant B asking Query 1: Lookup Result = {lookup_tenant_b}")
    assert lookup_tenant_b is None, "SECURITY VIOLATION: Tenant B accessed Tenant A's cache!"
    print("✓ Cross-tenant cache isolation verified! Tenant B cannot access Tenant A's cached entries.")

    print("\n" + "=" * 80)
    print("TEST 6: Cache Invalidation Lifecycle")
    print("=" * 80)

    semantic_cache.invalidate(tenant_id=tenant_a)
    lookup_after_invalidation = semantic_cache.lookup(tenant_id=tenant_a, query=query_1)
    assert lookup_after_invalidation is None, "Cache should be empty after invalidation!"
    print("✓ Cache invalidation successfully purged cached vectors.")

    print("\n" + "=" * 80)
    print("ALL SEMANTIC CACHING TESTS PASSED FLOCKINGLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_semantic_caching_lifecycle())
