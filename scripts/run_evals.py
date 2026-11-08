import os
import sys
import argparse

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath("."))

from evals.runner import MatrixEvalRunner
from evals.matrix import MatrixAggregator


def main():
    parser = argparse.ArgumentParser(description="DocuMind 2D Document-Question Type Evaluation Matrix Runner")
    parser.add_argument("--max-cases", type=int, default=None, help="Maximum number of test cases to run")
    parser.add_argument("--metric", type=str, default="composite_score", choices=MatrixAggregator.METRIC_NAMES, help="Metric to display in matrix table")
    args = parser.parse_args()

    runner = MatrixEvalRunner()
    report = runner.run_all(max_cases=args.max_cases)
    agg = report["aggregated_matrix"]
    overall = agg["overall_summary"]

    print("\n" + "=" * 80)
    print("DOCUMIND ENTERPRISE 2D EVALUATION MATRIX RESULTS")
    print("=" * 80)

    # 1. Executive Summary Table
    print("\n### System-Wide Performance Summary across All Enterprise Metrics:")
    print(f"| Metric Name | Evaluation Scope | Overall Score | Target Benchmark |")
    print(f"| :--- | :--- | :---: | :---: |")
    print(f"| **Recall@1** | Top-1 retrieval accuracy | **{overall.get('recall_1', 0.0):.2f}** | >= 0.70 |")
    print(f"| **Recall@3** | Top-3 retrieval coverage | **{overall.get('recall_3', 0.0):.2f}** | >= 0.85 |")
    print(f"| **Recall@5** | Top-5 retrieval coverage | **{overall.get('recall_5', 0.0):.2f}** | >= 0.90 |")
    print(f"| **Precision@3** | Relevance density in top 3 | **{overall.get('precision_3', 0.0):.2f}** | >= 0.50 |")
    print(f"| **MRR** | Mean Reciprocal Rank (1/rank) | **{overall.get('mrr', 0.0):.2f}** | >= 0.75 |")
    print(f"| **NDCG@3** | Normalized Discounted Cumulative Gain | **{overall.get('ndcg_3', 0.0):.2f}** | >= 0.80 |")
    print(f"| **Faithfulness** | Claim-level context entailment (zero hallucination) | **{overall.get('faithfulness', 0.0):.2f}** | >= 0.95 |")
    print(f"| **Answer Relevancy** | Query intent satisfaction & conciseness | **{overall.get('relevance', 0.0):.2f}** | >= 0.85 |")
    print(f"| **Refusal Accuracy** | Negative rejection & zero false refusal | **{overall.get('refusal_accuracy', 0.0):.2f}** | >= 0.90 |")
    print(f"| **Composite Quality** | Balanced Retrieval & Generation Score | **{overall.get('composite_score', 0.0):.2f}** | >= 0.85 |")

    print("\n" + "=" * 80)
    md_primary = MatrixAggregator.format_markdown_table(agg, metric=args.metric, title=f"### 2D Matrix: `{args.metric}`")
    print(md_primary)
    print("\n" + "=" * 80)

    # Write complete multi-metric scorecard artifact to disk
    os.makedirs("evals/results", exist_ok=True)
    scorecard_md_path = "evals/results/latest_scorecard.md"

    with open(scorecard_md_path, "w", encoding="utf-8") as f:
        f.write("# DocuMind Enterprise Comprehensive 2D Evaluation Scorecard\n\n")
        f.write(f"**Evaluated At**: {report['timestamp']} UTC  \n")
        f.write(f"**Total Test Cases Evaluated**: {report['total_evaluated']} across 49 Matrix Cells  \n")
        f.write(f"**Target Tenant ID**: `{report['tenant_id']}`  \n\n")

        f.write("## 1. Executive Summary: All 7 Enterprise Metrics\n\n")
        f.write("| Metric | Stage | Formula / Definition | Corpus Overall Score |\n")
        f.write("| :--- | :---: | :--- | :---: |\n")
        f.write(f"| **Recall@3** | Retrieval | Ground-truth chunk present in top 3 results | **{overall.get('recall_3', 0.0):.2f}** |\n")
        f.write(f"| **Precision@3** | Retrieval | Fraction of top 3 chunks containing ground-truth evidence | **{overall.get('precision_3', 0.0):.2f}** |\n")
        f.write(f"| **MRR** | Retrieval | Mean Reciprocal Rank ($1/\\text{{rank}}$ of first relevant chunk) | **{overall.get('mrr', 0.0):.2f}** |\n")
        f.write(f"| **NDCG@3** | Retrieval | Normalized Discounted Cumulative Gain with logarithmic rank discount | **{overall.get('ndcg_3', 0.0):.2f}** |\n")
        f.write(f"| **Faithfulness** | Generation | Ratio of answer claims strictly entailed by context (Hallucination metric) | **{overall.get('faithfulness', 0.0):.2f}** |\n")
        f.write(f"| **Answer Relevancy** | Generation | Semantic alignment and completeness of answer to user query | **{overall.get('relevance', 0.0):.2f}** |\n")
        f.write(f"| **Refusal Accuracy** | Generation | Rejection of unanswerable queries without false refusal of answerable ones | **{overall.get('refusal_accuracy', 0.0):.2f}** |\n\n")

        f.write("## 2. Retrieval Evaluation Matrices\n\n")
        f.write(MatrixAggregator.format_markdown_table(agg, metric="recall_3", title="### Metric: Recall@3 (Top-3 Retrieval Coverage)") + "\n\n")
        f.write(MatrixAggregator.format_markdown_table(agg, metric="precision_3", title="### Metric: Precision@3 (Top-3 Chunk Relevance Density)") + "\n\n")
        f.write(MatrixAggregator.format_markdown_table(agg, metric="mrr", title="### Metric: MRR (Mean Reciprocal Rank)") + "\n\n")
        f.write(MatrixAggregator.format_markdown_table(agg, metric="ndcg_3", title="### Metric: NDCG@3 (Normalized Discounted Cumulative Gain)") + "\n\n")

        f.write("## 3. Generation & LLM-as-a-Judge Matrices\n\n")
        f.write(MatrixAggregator.format_markdown_table(agg, metric="faithfulness", title="### Metric: Faithfulness (Hallucination Resistance)") + "\n\n")
        f.write(MatrixAggregator.format_markdown_table(agg, metric="relevance", title="### Metric: Answer Relevancy") + "\n\n")
        f.write(MatrixAggregator.format_markdown_table(agg, metric="refusal_accuracy", title="### Metric: Refusal Accuracy") + "\n\n")

        f.write("## 4. Composite Quality Matrix\n\n")
        f.write(MatrixAggregator.format_markdown_table(agg, metric="composite_score", title="### Metric: Composite Quality Score") + "\n\n")

    print(f"Comprehensive multi-metric scorecard saved to: {scorecard_md_path}\n")


if __name__ == "__main__":
    main()
