"""Build the condition 4 fine-tuning set from the Spider train split.

Training prompts are produced by the same `build_prompt` used at inference, with
no retrieved examples and no linked values -- so they are byte-identical to the
condition 2 prompt the adapter will be served with.

Condition 5 will add a value block at inference that the adapter never sees here.
That is deliberate: one adapter, RAG added at serve time, is what "fine-tuned
model + RAG" means in the brief, and training on the value format instead would
put condition 4 out of distribution rather than condition 5.
"""

from rag_text2sql.data import format_schema_prompt
from rag_text2sql.models.cloud import SYSTEM_PROMPT
from rag_text2sql.rag import build_prompt


def build_messages(question: str, schema: dict, gold_sql: str) -> list[dict]:
    """One chat-format SFT example: the served prompt, answered by the gold query.

    The trailing semicolon 387 of the 7,000 train queries carry is stripped, since
    SYSTEM_PROMPT tells the model not to emit one -- training targets that keep it
    would teach the opposite of the instruction it is served with.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(format_schema_prompt(schema), question, [])},
        {"role": "assistant", "content": gold_sql.strip().rstrip(";").strip()},
    ]
