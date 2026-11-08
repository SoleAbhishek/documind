import json
import re
from typing import Dict, Any, List, Optional
from src.config import settings


class GenerationEvaluator:
    """
    LLM-as-a-Judge evaluation engine for generation quality:
    1. Faithfulness (Claim-by-claim context entailment)
    2. Answer Relevance (Directness and adherence to the user query)
    3. Refusal Correctness (Accurate detection of unanswerable questions)
    """

    def __init__(self):
        self._client = None
        self._model_name = settings.GEMINI_CHAT_MODEL.replace("models/", "")

    def _get_client(self):
        if self._client is None:
            from google import genai
            api_key = settings.GEMINI_API_KEY
            if not api_key or api_key.startswith("AIzaSy-your") or len(api_key) < 20:
                raise ValueError("Valid GEMINI_API_KEY is required to run LLM-as-a-Judge evaluations.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def evaluate_unanswerable(self, generated_answer: str) -> float:
        """
        Evaluates whether an unanswerable question was correctly refused without hallucination.
        """
        refusal_phrases = [
            "cannot find sufficient information",
            "not mentioned in the provided",
            "not found in the indexed",
            "does not contain information",
            "insufficient information",
            "no information",
            "unable to find"
        ]
        lower_ans = generated_answer.lower()
        for phrase in refusal_phrases:
            if phrase in lower_ans:
                return 1.0
        return 0.0

    def evaluate_faithfulness_and_relevance(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        ground_truth_answer: str
    ) -> Dict[str, Any]:
        """
        Uses Gemini LLM judge to evaluate:
        - Faithfulness (0.0 to 1.0)
        - Answer Relevance (0.0 to 1.0)
        - Correctness against Ground Truth (0.0 to 1.0)
        """
        client = self._get_client()

        context_text = "\n\n---\n\n".join(retrieved_contexts[:4])

        judge_prompt = f"""You are an expert AI Evaluation Judge specializing in Retrieval-Augmented Generation (RAG) benchmarks.
Carefully evaluate the following generated answer against the provided context and ground truth.

User Question:
{query}

Retrieved Context:
{context_text}

Generated Answer to Evaluate:
{generated_answer}

Expected Ground Truth:
{ground_truth_answer}

Instructions:
1. Faithfulness (Groundedness):
   - Break down the generated answer into its core factual claims.
   - For each claim, check if it is directly supported by the Retrieved Context.
   - Calculate faithfulness_score = (supported claims) / (total claims).
   - If the answer is an explicit refusal ("I cannot find..."), give faithfulness_score = 1.0.
   - Score must be between 0.0 and 1.0.

2. Answer Relevance:
   - Does the generated answer directly address what the user asked without irrelevant filler?
   - Score must be between 0.0 and 1.0.

3. Semantic Correctness:
   - Does the generated answer convey the same factual meaning as the Expected Ground Truth?
   - Score must be between 0.0 and 1.0.

Output MUST be valid JSON strictly adhering to this schema:
{{
  "faithfulness_score": float,
  "relevance_score": float,
  "correctness_score": float,
  "reasoning": string
}}
"""
        import time
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=judge_prompt
                )
                raw_text = response.text.strip()
                # Clean markdown codeblocks if wrapped in ```json
                cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
                cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()

                result = json.loads(cleaned)
                return {
                    "faithfulness": float(result.get("faithfulness_score", 0.0)),
                    "relevance": float(result.get("relevance_score", 0.0)),
                    "correctness": float(result.get("correctness_score", 0.0)),
                    "judge_reasoning": result.get("reasoning", "")
                }
            except Exception as e:
                if ("429" in str(e) or "ResourceExhausted" in str(e)) and attempt < 2:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                return {
                    "faithfulness": 0.5,
                    "relevance": 0.5,
                    "correctness": 0.5,
                    "judge_reasoning": f"Evaluation fallback due to error: {e}"
                }
