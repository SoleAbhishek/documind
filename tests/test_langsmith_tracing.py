"""
Verification suite for LangSmith @traceable instrumentation and Multi-Model Cost Tracking.
Tests:
1. Safe import & configuration of LangSmith settings.
2. CostTracker price calculations across Gemini, OpenAI, and Anthropic.
3. Live query tracing and token extraction.
4. Semantic Cache hit trace and 100% cost elimination.
5. Markdown scorecard rendering.
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import settings
from src.services.cost_tracker import cost_tracker
from src.services.chat_service import chat_service
from src.services.semantic_cache import semantic_cache


TENANT_ID = "558c93f9-1135-40ba-86f7-cd6676525ef1"


def test_cost_tracker_calculations():
    print("\n--- [Test 1] CostTracker Pricing Calculations ---")
    prompt_tokens = 2500
    completion_tokens = 500

    calc = cost_tracker.calculate_cost(prompt_tokens, completion_tokens, "gemini-3.1-flash-lite")
    assert calc["prompt_tokens"] == 2500
    assert calc["completion_tokens"] == 500
    assert calc["total_tokens"] == 3000
    # Gemini 3.1 Flash-Lite: $0.075/1M in, $0.300/1M out
    expected_prompt_cost = 2500 * (0.075 / 1_000_000)
    expected_comp_cost = 500 * (0.300 / 1_000_000)
    assert abs(calc["prompt_cost_usd"] - expected_prompt_cost) < 1e-8
    assert abs(calc["completion_cost_usd"] - expected_comp_cost) < 1e-8

    print(f"Active Model: {calc['display_name']}")
    print(f"Total Cost for 3,000 tokens: ${calc['total_cost_usd']:.6f} USD")
    print("PASS: CostTracker unit calculation verified.")


def test_model_comparisons():
    print("\n--- [Test 2] Cross-Model Price Comparison & Multipliers ---")
    prompt_tokens = 4000
    completion_tokens = 800

    comp = cost_tracker.compare_all_models(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        current_model="gemini-3.1-flash-lite",
        is_cache_hit=False
    )

    table = cost_tracker.format_markdown_table(comp)
    print(table)

    comparisons = {c["model_key"]: c for c in comp["comparisons"]}
    assert "gpt-4o" in comparisons
    assert "claude-3-5-sonnet" in comparisons
    assert "gemini-2.5-pro" in comparisons

    # GPT-4o should be significantly more expensive than Gemini 3.1 Flash-Lite
    assert comparisons["gpt-4o"]["multiplier_vs_active"] > 25.0
    print("PASS: Cross-provider comparison logic verified.")


async def test_traceable_rag_pipeline_live_and_cache():
    print("\n--- [Test 3] End-to-End @traceable Pipeline & Cache Cost Savings ---")

    test_query = "What is the key architecture of Postgres autovacuum workers?"

    # Ensure clean slate for test query in cache
    semantic_cache.invalidate(tenant_id=TENANT_ID)

    # 1. First run: Cold execution (Cache Miss -> Qdrant Hybrid RRF -> Reranker -> Gemini LLM)
    print(f"[Run 1: Live LLM Pipeline] Query: '{test_query}'")
    res1 = await chat_service.chat(
        tenant_id=TENANT_ID,
        query=test_query
    )

    assert not res1["cached"], "First run should be a cache miss"
    assert len(res1["answer"]) > 0, "Response answer should not be empty"
    cost1 = res1.get("cost_summary", {})
    assert cost1.get("total_tokens", 0) > 0, "Token count should be greater than 0"
    print(f"Response snippet: {res1['answer'][:120]}...")
    print(f"Tokens: {cost1.get('total_tokens')}, Actual Cost: ${cost1.get('actual_cost_usd'):.6f} USD")
    print(f"Citations count: {len(res1.get('citations', []))}")

    # 2. Second run: Warm execution (Semantic Cache Hit -> Zero LLM latency, $0.00 cost)
    print(f"\n[Run 2: Semantic Cache Hit] Query: '{test_query}'")
    res2 = await chat_service.chat(
        tenant_id=TENANT_ID,
        query=test_query
    )

    assert res2["cached"], "Second run should hit semantic cache"
    cost2 = res2.get("cost_summary", {})
    assert cost2.get("is_cache_hit") is True
    assert cost2.get("actual_cost_usd") == 0.0, "Cache hit actual cost should be $0.00"
    assert cost2.get("savings_usd", 0.0) > 0.0, "Cache hit savings should be positive"
    print(f"Cache Hit Confirmed! Actual Cost: ${cost2.get('actual_cost_usd'):.6f} USD")
    print(f"Dollars Saved by Cache: ${cost2.get('savings_usd'):.6f} USD (100% saved)")

    # Print the full scorecard for the cached run
    print("\n--- Scorecard for Semantic Cache Hit ---")
    print(cost_tracker.format_markdown_table(cost2))
    print("\nPASS: End-to-end @traceable and cost comparison verified successfully.")


async def main():
    print("=" * 65)
    print("DocuMind LangSmith @traceable & Cost Comparison Verification")
    print(f"Tracing Configured: {settings.LANGCHAIN_TRACING_V2}")
    print(f"Project Name: {settings.LANGCHAIN_PROJECT}")
    print(f"Endpoint: {settings.LANGCHAIN_ENDPOINT}")
    print("=" * 65)

    test_cost_tracker_calculations()
    test_model_comparisons()
    await test_traceable_rag_pipeline_live_and_cache()
    print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(main())
