from typing import List

from interface.base_response_generator import BaseResponseGenerator
from util.invoke_ai import invoke_ai


SYSTEM_MESSAGE = """You are a precise question-answering assistant.
You will be given a user question and a set of context passages retrieved from a document corpus.

Rules you MUST follow:
1. Answer ONLY using information found in the provided context.
2. If the answer is not present in the context, respond with exactly:
   "I don't have enough information to answer this question."
3. Do NOT make up facts, infer beyond what is stated, or use external knowledge.
4. Keep your answer concise and directly address the question.
5. If relevant, cite which part of the context supports your answer.
"""


class ResponseGenerator(BaseResponseGenerator):

    def generate_response(self, query: str, context: List[str]) -> str:
        if not context:
            return "I don't have enough information to answer this question."

        formatted_context = "\n\n".join(
            f"[Passage {i + 1}]:\n{chunk}" for i, chunk in enumerate(context)
        )

        user_message = f"""Context:
            {formatted_context}

            Question: {query}

            Answer:"""

        return invoke_ai(system_message=SYSTEM_MESSAGE, user_message=user_message)
