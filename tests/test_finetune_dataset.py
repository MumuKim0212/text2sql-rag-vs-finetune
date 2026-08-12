from pathlib import Path

from rag_text2sql.data import format_schema_prompt, load_schemas
from rag_text2sql.finetune.dataset import build_messages
from rag_text2sql.models.cloud import SYSTEM_PROMPT

TABLES_JSON = Path("data/spider/tables.json")


def _schema():
    return load_schemas(TABLES_JSON)["concert_singer"]


def test_training_prompt_matches_the_served_condition_2_prompt():
    """The adapter must be trained on exactly the prompt it is served with."""
    schema = _schema()
    question = "How many singers are there?"

    messages = build_messages(question, schema, "SELECT count(*) FROM singer")

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == (
        f"Schema:\n{format_schema_prompt(schema)}\n\nQuestion: {question}\nSQL:"
    )


def test_gold_target_drops_the_trailing_semicolon():
    """SYSTEM_PROMPT forbids one, so targets carrying it would train against the instruction."""
    messages = build_messages("q?", _schema(), "SELECT count(*) FROM singer;")

    assert messages[2]["content"] == "SELECT count(*) FROM singer"
