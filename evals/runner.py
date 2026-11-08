import os
import sys
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from src.config import settings
from src.search.vector_store import vector_store
from src.services.chat_service import ChatService
from evals.evaluators.retrieval import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k
)
from evals.evaluators.generation import GenerationEvaluator
from evals.matrix import MatrixAggregator


class MatrixEvalRunner:
    """
    Executes the end-to-end evaluation suite across the 2D Document-Question Matrix.
    """

    def __init__(
        self,
        dataset_path: str = "evals/datasets/golden_eval_dataset.json",
        tenant_id: Optional[str] = None
    ):
        self.dataset_path = dataset_path
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.tenant_id = tenant_id or self.dataset.get("target_tenant_id", "558c93f9-1135-40ba-86f7-cd6676525ef1")
        self.chat_service = ChatService()
        self.gen_evaluator = GenerationEvaluator()

    def generate_rag_answer(self, query: str, top_chunks: List[Dict[str, Any]]) -> str:
        """
        Executes non-streaming Gemini generation using DocuMind's grounded system prompt.
        """
        system_prompt = self.chat_service._build_system_prompt(top_chunks)
        client = self.chat_service._get_genai_client()
        if not client:
            return "Error: Gemini client could not be initialized."

        model_name = settings.GEMINI_CHAT_MODEL.replace("models/", "")
        combined_prompt = f"{system_prompt}\n\nUser Question:\n{query}"

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=combined_prompt
                )
                return response.text.strip()
            except Exception as e:
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    time.sleep(3.0 * (attempt + 1))
                    continue
                return f"Generation error: {e}"
        return "Generation error: Max retries exceeded"

    def run_single_test(self, tc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single test case through both Retrieval and Generation stages.
        """
        query = tc["query"]
        expected_keywords = tc.get("expected_keywords", [])
        ground_truth = tc.get("ground_truth_answer", "")
        is_unans = tc.get("is_unanswerable", False)
        doc_id = tc.get("document_id")

        print(f"\nEvaluating [{tc['id']}] ({tc['document_category']} / {tc['question_type']}):", flush=True)
        print(f"  Query: '{query[:80]}...'", flush=True)

        # 1. Retrieval Stage (Hybrid Search)
        start_retrieval = time.time()
        retrieved_raw = vector_store.search_hybrid(
            tenant_id=self.tenant_id,
            query=query,
            limit=10,
            document_id=doc_id
        )
        retrieval_ms = round((time.time() - start_retrieval) * 1000, 2)

        # Compute retrieval metrics (Recall, Precision, MRR, NDCG)
        recall_1 = compute_recall_at_k(retrieved_raw, expected_keywords, k=1)
        recall_3 = compute_recall_at_k(retrieved_raw, expected_keywords, k=3)
        recall_5 = compute_recall_at_k(retrieved_raw, expected_keywords, k=5)

        precision_1 = compute_precision_at_k(retrieved_raw, expected_keywords, k=1)
        precision_3 = compute_precision_at_k(retrieved_raw, expected_keywords, k=3)
        precision_5 = compute_precision_at_k(retrieved_raw, expected_keywords, k=5)

        mrr = compute_mrr(retrieved_raw, expected_keywords)
        ndcg_3 = compute_ndcg_at_k(retrieved_raw, expected_keywords, k=3)
        ndcg_5 = compute_ndcg_at_k(retrieved_raw, expected_keywords, k=5)

        # 2. Rerank Stage
        top_chunks = self.chat_service.rerank_chunks(query, retrieved_raw, top_k=4)
        retrieved_texts = [c.get("text", "") for c in top_chunks]

        # 3. Generation Stage
        start_gen = time.time()
        generated_answer = self.generate_rag_answer(query, top_chunks)
        gen_ms = round((time.time() - start_gen) * 1000, 2)

        # 4. Evaluation Stage
        refusal_detected = self.gen_evaluator.evaluate_unanswerable(generated_answer)

        if is_unans:
            # For unanswerables: Refusal is expected.
            refusal_accuracy = 1.0 if refusal_detected > 0.5 else 0.0
            faithfulness = 1.0 if refusal_detected > 0.5 else 0.0
            relevance = 1.0 if refusal_detected > 0.5 else 0.0
            correctness = 1.0 if refusal_detected > 0.5 else 0.0
            reasoning = f"Unanswerable test: Refusal detected = {refusal_detected}"
        else:
            # For answerable: Falsely refusing is penalized in refusal_accuracy
            refusal_accuracy = 0.0 if refusal_detected > 0.5 else 1.0
            gen_metrics = self.gen_evaluator.evaluate_faithfulness_and_relevance(
                query=query,
                retrieved_contexts=retrieved_texts,
                generated_answer=generated_answer,
                ground_truth_answer=ground_truth
            )
            faithfulness = gen_metrics["faithfulness"]
            relevance = gen_metrics["relevance"]
            correctness = gen_metrics["correctness"]
            reasoning = gen_metrics["judge_reasoning"]

        result = {
            "id": tc["id"],
            "document_category": tc["document_category"],
            "question_type": tc["question_type"],
            "document_name": tc["document_name"],
            "query": query,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth,
            "recall_1": recall_1,
            "recall_3": recall_3,
            "recall_5": recall_5,
            "precision_1": precision_1,
            "precision_3": precision_3,
            "precision_5": precision_5,
            "mrr": round(mrr, 3),
            "ndcg_3": round(ndcg_3, 3),
            "ndcg_5": round(ndcg_5, 3),
            "faithfulness": round(faithfulness, 2),
            "relevance": round(relevance, 2),
            "refusal_accuracy": round(refusal_accuracy, 2),
            "correctness": round(correctness, 2),
            "judge_reasoning": reasoning,
            "retrieval_ms": retrieval_ms,
            "generation_ms": gen_ms
        }

        print(f"  Recall@3: {recall_3} | MRR: {mrr:.2f} | Faithfulness: {faithfulness} | Relevance: {relevance}", flush=True)
        return result

    def run_all(self, max_cases: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs all test cases and computes the full 2D evaluation matrix.
        """
        test_cases = self.dataset["test_cases"]
        if max_cases:
            test_cases = test_cases[:max_cases]

        print(f"[DocuMind Evals] Launching evaluation of {len(test_cases)} test cases across 2D Matrix...", flush=True)

        results = []
        for tc in test_cases:
            res = self.run_single_test(tc)
            results.append(res)
            time.sleep(1.0)  # Gentle pacing for LLM API calls

        aggregated = MatrixAggregator.aggregate(results)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": timestamp,
            "tenant_id": self.tenant_id,
            "total_evaluated": len(results),
            "aggregated_matrix": aggregated,
            "detailed_results": results
        }

        # Save to disk
        os.makedirs("evals/results", exist_ok=True)
        report_path = f"evals/results/matrix_eval_report_{timestamp}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n[DocuMind Evals] Completed successfully! Saved full report to: {report_path}")
        return report
