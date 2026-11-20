"""
DocuMind Token Usage & Model Cost Tracking Service.
Calculates query execution costs, compares pricing across Google Gemini, OpenAI,
and Anthropic models, and quantifies cost savings achieved via Semantic Caching.
"""

from typing import Dict, Any, Optional
from src.config import settings


# Pricing Catalog per 1,000,000 tokens (USD)
# Updated to current public API pricing (2025/2026)
MODEL_PRICING_CATALOG: Dict[str, Dict[str, Any]] = {
    # Google Gemini Models
    "gemini-3.1-flash-lite": {
        "provider": "Google",
        "display_name": "Gemini 3.1 Flash-Lite (Active Default)",
        "input_cost_per_1m": 0.075,
        "output_cost_per_1m": 0.300,
    },
    "gemini-2.5-flash": {
        "provider": "Google",
        "display_name": "Gemini 2.5 Flash",
        "input_cost_per_1m": 0.075,
        "output_cost_per_1m": 0.300,
    },
    "gemini-1.5-flash": {
        "provider": "Google",
        "display_name": "Gemini 1.5 Flash",
        "input_cost_per_1m": 0.075,
        "output_cost_per_1m": 0.300,
    },
    "gemini-2.5-pro": {
        "provider": "Google",
        "display_name": "Gemini 2.5 Pro",
        "input_cost_per_1m": 1.250,
        "output_cost_per_1m": 5.000,
    },
    "gemini-1.5-pro": {
        "provider": "Google",
        "display_name": "Gemini 1.5 Pro",
        "input_cost_per_1m": 1.250,
        "output_cost_per_1m": 5.000,
    },
    # OpenAI Counterpart Models
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "display_name": "GPT-4o Mini",
        "input_cost_per_1m": 0.150,
        "output_cost_per_1m": 0.600,
    },
    "gpt-4o": {
        "provider": "OpenAI",
        "display_name": "GPT-4o",
        "input_cost_per_1m": 2.500,
        "output_cost_per_1m": 10.000,
    },
    # Anthropic Counterpart Models
    "claude-3-5-sonnet": {
        "provider": "Anthropic",
        "display_name": "Claude 3.5 Sonnet",
        "input_cost_per_1m": 3.000,
        "output_cost_per_1m": 15.000,
    },
}


class CostTracker:
    """
    Real-time cost calculator and model pricing comparison engine.
    """

    def __init__(self, catalog: Optional[Dict[str, Dict[str, Any]]] = None):
        self.catalog = catalog or MODEL_PRICING_CATALOG

    def estimate_tokens(self, text: str) -> int:
        """
        Estimates token count using character and word heuristics
        when API usage metadata is unavailable.
        Rule of thumb: ~4 characters per token in English.
        """
        if not text:
            return 0
        char_count = len(text)
        word_count = len(text.split())
        # Average between character-based (4 chars/token) and word-based (0.75 words/token)
        return max(1, int((char_count / 4.0 + word_count * 1.33) / 2.0))

    def _normalize_model_name(self, model_name: Optional[str]) -> str:
        """Maps model name variations to catalog keys."""
        if not model_name:
            model_name = settings.GEMINI_CHAT_MODEL
        model_name = model_name.replace("models/", "").lower()
        if model_name in self.catalog:
            return model_name
        # Fallback matching
        for key in self.catalog:
            if key in model_name or model_name in key:
                return key
        return "gemini-3.1-flash-lite"

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates exact dollar cost for a specific model based on token usage.
        """
        key = self._normalize_model_name(model_name)
        rates = self.catalog.get(key, self.catalog["gemini-3.1-flash-lite"])

        input_rate = rates["input_cost_per_1m"] / 1_000_000.0
        output_rate = rates["output_cost_per_1m"] / 1_000_000.0

        prompt_cost = prompt_tokens * input_rate
        completion_cost = completion_tokens * output_rate
        total_cost = prompt_cost + completion_cost

        return {
            "model_key": key,
            "provider": rates["provider"],
            "display_name": rates["display_name"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_cost_usd": prompt_cost,
            "completion_cost_usd": completion_cost,
            "total_cost_usd": total_cost,
        }

    def compare_all_models(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        current_model: Optional[str] = None,
        is_cache_hit: bool = False
    ) -> Dict[str, Any]:
        """
        Generates cross-provider cost comparison and semantic cache savings.
        """
        curr_key = self._normalize_model_name(current_model)
        active_calc = self.calculate_cost(prompt_tokens, completion_tokens, curr_key)
        active_cost = active_calc["total_cost_usd"]

        # Cache calculation: if hit, actual cost is zero
        actual_cost = 0.0 if is_cache_hit else active_cost
        savings_usd = active_cost if is_cache_hit else 0.0
        savings_pct = 100.0 if is_cache_hit else 0.0

        comparison_list = []
        # Representative models to compare against
        target_keys = [
            "gemini-3.1-flash-lite",
            "gemini-2.5-pro",
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-5-sonnet",
        ]

        for k in target_keys:
            if k not in self.catalog:
                continue
            calc = self.calculate_cost(prompt_tokens, completion_tokens, k)
            cost = calc["total_cost_usd"]
            mult = (cost / active_cost) if active_cost > 0 else 1.0

            comparison_list.append({
                "model_key": k,
                "display_name": calc["display_name"],
                "provider": calc["provider"],
                "total_cost_usd": cost,
                "multiplier_vs_active": round(mult, 2),
                "is_active": (k == curr_key),
            })

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "is_cache_hit": is_cache_hit,
            "active_model": curr_key,
            "actual_cost_usd": actual_cost,
            "nominal_cost_usd": active_cost,
            "savings_usd": savings_usd,
            "savings_pct": savings_pct,
            "comparisons": comparison_list,
        }

    def format_markdown_table(self, comparison: Dict[str, Any]) -> str:
        """
        Generates a clean Markdown pricing scorecard.
        """
        is_hit = comparison.get("is_cache_hit", False)
        tokens_in = comparison.get("prompt_tokens", 0)
        tokens_out = comparison.get("completion_tokens", 0)
        actual = comparison.get("actual_cost_usd", 0.0)

        lines = [
            f"### Model Pricing & Cost Comparison ({tokens_in:,} in / {tokens_out:,} out tokens)",
            f"- **Semantic Cache Hit**: {'Yes (⚡ 100% cost eliminated)' if is_hit else 'No (Live LLM execution)'}",
            f"- **Actual Query Cost**: `${actual:.6f} USD`",
            "",
            "| Model | Provider | Query Cost (USD) | Relative Multiplier | Active |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for c in comparison.get("comparisons", []):
            active_marker = " Active" if c.get("is_active") else ""
            cost_str = f"${c['total_cost_usd']:.6f}"
            mult_str = f"{c['multiplier_vs_active']:.1f}x"
            lines.append(f"| {c['display_name']} | {c['provider']} | {cost_str} | {mult_str} | {active_marker} |")

        if is_hit:
            saved = comparison.get("savings_usd", 0.0)
            lines.append("")
            lines.append(f"> **Cache Benefit**: Saved `${saved:.6f} USD` on this query by serving directly from Qdrant semantic vector cache.")

        return "\n".join(lines)


# Global singleton instance
cost_tracker = CostTracker()
