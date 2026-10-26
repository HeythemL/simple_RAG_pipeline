import os
from groq import Groq


def invoke_ai(system_message: str, user_message: str) -> str:

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content