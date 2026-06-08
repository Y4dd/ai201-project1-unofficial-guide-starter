import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['source_file']}\n{chunk['text']}")
    return "\n\n".join(parts)


def _system_prompt(context: str) -> str:
    return f"""You are a housing advisor for Western Michigan University students.

Answer the student's question using ONLY the information provided in the context below.
Do not use any knowledge from outside these documents.
If the context does not contain enough information to answer the question, respond with exactly:
"I don't have enough information in my documents to answer that question."

At the end of every answer, add a "Sources:" line listing the source filenames you used
(e.g., "Sources: source1_wmu_residence_halls.md, source6_winter_commuting.md").

CONTEXT:
{context}"""


def generate(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have enough information in my documents to answer that question."

    context = _format_context(chunks)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt(context)},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content
