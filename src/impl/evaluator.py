import json
import os
from typing import List

from groq import Groq
from pydantic import BaseModel

from interface.base_evaluator import BaseEvaluator, EvaluationResult

SAMPLE_QUESTIONS_PATH = "sample_data/evaluation/sample_questions.json"

SYSTEM_PROMPT = """You are a strict but fair judge evaluating whether a RAG pipeline's answer is correct.

You will be given:
- A question
- The pipeline's response
- The expected correct answer

Your job is to decide if the response is CORRECT or INCORRECT.

Rules:
- Mark as CORRECT if the response conveys the same key facts as the expected answer (wording can differ).
- Mark as INCORRECT if the response is missing critical facts, contains wrong information, or says it doesn't know when an answer exists.
- Be concise in your reasoning (1-2 sentences max).

Respond ONLY with valid JSON in this exact format:
{
  "is_correct": true or false,
  "reasoning": "brief explanation"
}"""


class Evaluator(BaseEvaluator):

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def evaluate(self, query: str, response: str, expected_answer: str) -> EvaluationResult:
        user_message = f"""Question: {query}

Pipeline Response: {response}

Expected Answer: {expected_answer}"""

        completion = self.client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        raw = completion.choices[0].message.content.strip()

        try:
            verdict = json.loads(raw)
            is_correct = bool(verdict["is_correct"])
            reasoning = verdict.get("reasoning", "")
        except (json.JSONDecodeError, KeyError):
            is_correct = False
            reasoning = f"Failed to parse judge response: {raw}"

        return EvaluationResult(
            question=query,
            response=response,
            expected_answer=expected_answer,
            is_correct=is_correct,
            reasoning=reasoning,
        )

    def run_evaluation(self, pipeline_fn) -> List[EvaluationResult]:
        """
        Run the full evaluation suite against the pipeline.

        Args:
            pipeline_fn: callable that takes a question (str) and returns an answer (str).

        Returns:
            List of EvaluationResult, one per sample question.
        """
        with open(SAMPLE_QUESTIONS_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f)

        results: List[EvaluationResult] = []

        for i, sample in enumerate(samples):
            question = sample["question"]
            expected = sample["answer"]

            print(f"[{i + 1}/{len(samples)}] Evaluating: {question[:60]}...")

            try:
                response = pipeline_fn(question)
            except Exception as e:
                response = f"[Pipeline error: {e}]"

            result = self.evaluate(
                query=question,
                response=response,
                expected_answer=expected,
            )
            results.append(result)

            status = "[OK]" if result.is_correct else "[FAIL]"
            print(f"  {status} {result.reasoning}")

        correct = sum(1 for r in results if r.is_correct)
        print(f"\nScore: {correct}/{len(results)} ({100 * correct // len(results)}%)")

        return results