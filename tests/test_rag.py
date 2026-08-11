from pathlib import Path

import numpy as np

from rag_text2sql.data import format_schema_prompt, load_schemas
from rag_text2sql.rag import build_prompt, prune_schema, table_texts, top_k

TABLES_JSON = Path("data/spider/tables.json")
WIDE_DB = "student_transcripts_tracking"  # 11 tables, the widest dev schema


def test_top_k_returns_best_matches_in_order():
    index = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    queries = np.array([[1.0, 0.0]])

    assert top_k(queries, index, k=2).tolist() == [[0, 2]]


def test_top_k_clamps_to_index_size():
    index = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert top_k(np.array([[1.0, 0.0]]), index, k=5).shape == (1, 2)


def test_table_texts_lists_columns_per_table():
    schema = load_schemas(TABLES_JSON)["concert_singer"]

    texts = table_texts(schema)

    assert len(texts) == len(schema["table_names_original"])
    assert texts[0].startswith("stadium: ")
    assert "*" not in "".join(texts)  # the table-less "*" column is not part of any table


def test_prune_schema_keeps_only_requested_tables():
    schema = load_schemas(TABLES_JSON)[WIDE_DB]
    keep = [1, 3, 0]

    pruned = prune_schema(schema, keep)

    assert pruned["table_names_original"] == [schema["table_names_original"][i] for i in sorted(keep)]
    rendered = format_schema_prompt(pruned)
    for i, name in enumerate(schema["table_names_original"]):
        assert (f'CREATE TABLE "{name}"' in rendered) == (i in keep)


def test_prune_schema_remaps_column_pk_and_fk_indices():
    """Columns, PKs and FKs are global indices, so pruning must renumber them all."""
    schema = load_schemas(TABLES_JSON)[WIDE_DB]
    kept_names = {schema["table_names_original"][i] for i in (2, 4)}

    pruned = prune_schema(schema, [2, 4])

    # Every surviving column still belongs to a kept table and names the same column.
    original_by_name = {
        (schema["table_names_original"][t], c) for t, c in schema["column_names_original"] if t != -1
    }
    for table_idx, col_name in pruned["column_names_original"]:
        if table_idx == -1:
            continue
        assert pruned["table_names_original"][table_idx] in kept_names
        assert (pruned["table_names_original"][table_idx], col_name) in original_by_name

    n_columns = len(pruned["column_names_original"])
    assert all(0 <= c < n_columns for c in pruned["primary_keys"])
    assert all(0 <= c < n_columns for pair in pruned["foreign_keys"] for c in pair)
    # FKs pointing at a dropped table are gone, not dangling.
    assert len(pruned["foreign_keys"]) <= len(schema["foreign_keys"])


def test_prune_schema_keeping_everything_matches_the_original_render():
    schema = load_schemas(TABLES_JSON)[WIDE_DB]

    pruned = prune_schema(schema, list(range(len(schema["table_names_original"]))))

    assert format_schema_prompt(pruned) == format_schema_prompt(schema)


def test_build_prompt_ends_with_the_no_rag_prompt():
    """Only the demonstrations block may differ between condition 2 and condition 3."""
    schema_prompt = 'CREATE TABLE "singer" (\n  "id" number\n)'
    question = "How many singers are there?"

    prompt = build_prompt(schema_prompt, question, [("How many cats?", "SELECT count(*) FROM cats")])

    assert prompt.endswith(f"Schema:\n{schema_prompt}\n\nQuestion: {question}\nSQL:")
    assert "Question: How many cats?\nSQL: SELECT count(*) FROM cats" in prompt
