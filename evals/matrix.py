from typing import List, Dict, Any
from collections import defaultdict


class MatrixAggregator:
    """
    Aggregates test results into a 2D Document Category x Question Type matrix.
    Computes cell-level averages and row/column marginals across all enterprise RAG metrics:
    - Recall@1, Recall@3, Recall@5
    - Precision@1, Precision@3, Precision@5
    - MRR (Mean Reciprocal Rank)
    - NDCG@3, NDCG@5 (Normalized Discounted Cumulative Gain)
    - Faithfulness (Claim Entailment)
    - Answer Relevancy
    - Refusal Accuracy (Refusal of unanswerable + non-refusal of answerable)
    """

    DOCUMENT_CATEGORIES = [
        "Text-heavy",
        "Structured",
        "Technical",
        "Long-form",
        "Visual",
        "Complex",
        "Mixed"
    ]

    QUESTION_TYPES = [
        "Factual",
        "Semantic",
        "Numerical",
        "Comparative",
        "Multi-hop",
        "Table-based",
        "Unanswerable"
    ]

    METRIC_NAMES = [
        "recall_1",
        "recall_3",
        "recall_5",
        "precision_1",
        "precision_3",
        "precision_5",
        "mrr",
        "ndcg_3",
        "ndcg_5",
        "faithfulness",
        "relevance",
        "refusal_accuracy",
        "composite_score"
    ]

    @staticmethod
    def _calc_averages(items: List[Dict[str, Any]]) -> Dict[str, float]:
        if not items:
            return {}
        n = len(items)
        r3 = sum(i.get("recall_3", 0.0) for i in items) / n
        faith = sum(i.get("faithfulness", 0.0) for i in items) / n
        rel = sum(i.get("relevance", 0.0) for i in items) / n

        return {
            "count": n,
            "recall_1": round(sum(i.get("recall_1", 0.0) for i in items) / n, 3),
            "recall_3": round(r3, 3),
            "recall_5": round(sum(i.get("recall_5", 0.0) for i in items) / n, 3),
            "precision_1": round(sum(i.get("precision_1", 0.0) for i in items) / n, 3),
            "precision_3": round(sum(i.get("precision_3", 0.0) for i in items) / n, 3),
            "precision_5": round(sum(i.get("precision_5", 0.0) for i in items) / n, 3),
            "mrr": round(sum(i.get("mrr", 0.0) for i in items) / n, 3),
            "ndcg_3": round(sum(i.get("ndcg_3", 0.0) for i in items) / n, 3),
            "ndcg_5": round(sum(i.get("ndcg_5", 0.0) for i in items) / n, 3),
            "faithfulness": round(faith, 3),
            "relevance": round(rel, 3),
            "refusal_accuracy": round(sum(i.get("refusal_accuracy", 0.0) for i in items) / n, 3),
            "composite_score": round((r3 + faith + rel) / 3.0, 3)
        }

    @staticmethod
    def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Organizes flat test results into matrix cells and computes overall metrics.
        """
        grid = defaultdict(lambda: defaultdict(list))
        for r in results:
            cat = r.get("document_category", "Unknown")
            qtype = r.get("question_type", "Unknown")
            grid[cat][qtype].append(r)

        matrix_cells = {}
        category_summaries = {}
        question_type_summaries = {}

        for cat in MatrixAggregator.DOCUMENT_CATEGORIES:
            matrix_cells[cat] = {}
            for qtype in MatrixAggregator.QUESTION_TYPES:
                items = grid[cat][qtype]
                matrix_cells[cat][qtype] = MatrixAggregator._calc_averages(items) if items else None

        # Row averages (Category)
        for cat in MatrixAggregator.DOCUMENT_CATEGORIES:
            all_cat_items = [r for r in results if r.get("document_category") == cat]
            if all_cat_items:
                category_summaries[cat] = MatrixAggregator._calc_averages(all_cat_items)

        # Column averages (Question Type)
        for qtype in MatrixAggregator.QUESTION_TYPES:
            all_q_items = [r for r in results if r.get("question_type") == qtype]
            if all_q_items:
                question_type_summaries[qtype] = MatrixAggregator._calc_averages(all_q_items)

        # Global overall summary
        overall_summary = MatrixAggregator._calc_averages(results)

        return {
            "matrix": matrix_cells,
            "category_summaries": category_summaries,
            "question_type_summaries": question_type_summaries,
            "overall_summary": overall_summary,
            "total_evaluated": len(results)
        }

    @staticmethod
    def format_markdown_table(aggregated: Dict[str, Any], metric: str, title: str = None) -> str:
        """
        Formats the 2D matrix for a single metric into a Markdown table.
        """
        q_headers = MatrixAggregator.QUESTION_TYPES
        header_row = "| Document Category | " + " | ".join(q_headers) + " | Row Avg |"
        separator_row = "| :--- | " + " | ".join([":---:"] * len(q_headers)) + " | :---: |"

        display_title = title or f"### 2D Document-Question Matrix: `{metric}`"
        lines = [display_title, "", header_row, separator_row]

        matrix = aggregated["matrix"]
        cat_summaries = aggregated["category_summaries"]

        for cat in MatrixAggregator.DOCUMENT_CATEGORIES:
            row_vals = []
            for q in q_headers:
                cell = matrix.get(cat, {}).get(q)
                if cell is not None and metric in cell:
                    row_vals.append(f"{cell[metric]:.2f}")
                else:
                    row_vals.append("-")

            cat_avg = cat_summaries.get(cat, {}).get(metric, 0.0)
            lines.append(f"| **{cat}** | " + " | ".join(row_vals) + f" | **{cat_avg:.2f}** |")

        # Add Column Average footer
        col_averages = []
        for q in q_headers:
            q_avg = aggregated["question_type_summaries"].get(q, {}).get(metric, 0.0)
            col_averages.append(f"**{q_avg:.2f}**")
        overall_val = aggregated["overall_summary"].get(metric, 0.0)
        footer_row = "| **Col Avg** | " + " | ".join(col_averages) + f" | **{overall_val:.2f}** |"
        lines.append(footer_row)

        return "\n".join(lines)
