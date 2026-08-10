"""Cloud API baseline: Claude generating SQL from a question + schema prompt."""

import anthropic

SYSTEM_PROMPT = (
    "You are a Text-to-SQL engine. Given a database schema and a question, "
    "output ONLY the SQL query that answers the question. "
    "No explanation, no markdown code fences, no trailing semicolon."
)

DEFAULT_MODEL = "claude-sonnet-5"


def generate_sql(
    question: str,
    schema_prompt: str,
    model: str = DEFAULT_MODEL,
    client: anthropic.Anthropic | None = None,
) -> str:
    """Ask Claude for the SQL query answering `question` against `schema_prompt`."""
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Schema:\n{schema_prompt}\n\nQuestion: {question}\nSQL:"}],
    )
    return response.content[0].text.strip()
