"""Cloud API baseline (alternate provider): Gemini generating SQL from a question + schema prompt."""

from google import genai
from google.genai import types

from rag_text2sql.models.cloud import SYSTEM_PROMPT

DEFAULT_MODEL = "gemini-3.6-flash"


def generate_sql(
    question: str,
    schema_prompt: str,
    model: str = DEFAULT_MODEL,
    client: genai.Client | None = None,
) -> str:
    """Ask Gemini for the SQL query answering `question` against `schema_prompt`."""
    client = client or genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=f"Schema:\n{schema_prompt}\n\nQuestion: {question}\nSQL:",
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text.strip()
