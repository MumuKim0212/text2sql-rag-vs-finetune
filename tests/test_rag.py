from pathlib import Path

import numpy as np

from rag_text2sql.data import format_schema_prompt, load_schemas
from rag_text2sql.rag import (
    build_prompt,
    load_db_values,
    match_values,
    prune_schema,
    table_texts,
    top_k,
)

TABLES_JSON = Path("data/spider/tables.json")
WIDE_DB = "student_transcripts_tracking"  # 11 tables, the widest dev schema
PETS_DB = Path("data/spider/database/pets_1/pets_1.sqlite")  # stores PetType as 'cat'/'dog'


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


def test_load_db_values_reads_text_columns_only():
    values = load_db_values(PETS_DB)

    assert ("Pets", "PetType", "cat") in values
    # PetAge is numeric, so no value should be harvested from it
    assert not [v for v in values if v[1] == "PetAge"]


def test_match_values_finds_the_literal_the_model_gets_wrong():
    """The question says "cat", the column stores 'cat' -- casing is the whole point."""
    values = load_db_values(PETS_DB)

    hits = match_values("Find the first name of students who have cat or dog pet.", values)

    assert ("Pets", "PetType", "cat") in hits
    assert ("Pets", "PetType", "dog") in hits


def test_match_values_requires_whole_words():
    values = [("Pets", "PetType", "cat")]

    assert match_values("How many pets are in each category?", values) == []
    assert match_values("How many cats?", values) == []  # plural is not the stored value
    assert match_values("Show me every cat.", values) == [("Pets", "PetType", "cat")]


def test_match_values_is_capped():
    values = [("t", "c", f"value{i:03d}") for i in range(50)]
    question = " ".join(v[2] for v in values)

    assert len(match_values(question, values, max_hits=20)) == 20


def test_build_prompt_without_values_is_unchanged():
    """3a's prompt must not shift when the value-linking parameter exists but is unused."""
    schema_prompt = 'CREATE TABLE "singer" (\n  "id" number\n)'
    examples = [("How many cats?", "SELECT count(*) FROM cats")]

    assert build_prompt(schema_prompt, "q?", examples) == build_prompt(
        schema_prompt, "q?", examples, None
    )


def test_build_prompt_renders_linked_values_before_the_question():
    schema_prompt = 'CREATE TABLE "Pets" (\n  "PetType" text\n)'

    prompt = build_prompt(
        schema_prompt, "who has a cat?", [("q", "SELECT 1")], [("Pets", "PetType", "cat")]
    )

    assert 'Pets.PetType = "cat"' in prompt
    assert prompt.index("Pets.PetType") < prompt.index("Question: who has a cat?")
    assert prompt.endswith("Question: who has a cat?\nSQL:")


def test_build_prompt_ends_with_the_no_rag_prompt():
    """Only the demonstrations block may differ between condition 2 and condition 3."""
    schema_prompt = 'CREATE TABLE "singer" (\n  "id" number\n)'
    question = "How many singers are there?"

    prompt = build_prompt(schema_prompt, question, [("How many cats?", "SELECT count(*) FROM cats")])

    assert prompt.endswith(f"Schema:\n{schema_prompt}\n\nQuestion: {question}\nSQL:")
    assert "Question: How many cats?\nSQL: SELECT count(*) FROM cats" in prompt
