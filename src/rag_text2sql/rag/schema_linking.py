"""Schema linking: keep only the tables a question plausibly needs, drop the rest.

Table-level pruning only. On Spider dev this is a no-op for most databases
(median 3 tables), so it can only change the prompt for the handful of wide
schemas -- `scripts/build_rag_prompts.py` reports how many examples it touched.
"""


def table_texts(schema: dict) -> list[str]:
    """One "table: col, col, ..." line per table, embedded as the linking index."""
    table_names = schema["table_names_original"]
    columns_by_table: list[list[str]] = [[] for _ in table_names]
    for table_idx, col_name in schema["column_names_original"]:
        if table_idx != -1:
            columns_by_table[table_idx].append(col_name)
    return [f"{name}: {', '.join(columns)}" for name, columns in zip(table_names, columns_by_table)]


def prune_schema(schema: dict, keep_tables: list[int]) -> dict:
    """Return a tables.json-shaped entry holding only `keep_tables`.

    Every column, primary key and foreign key in tables.json is a *global column
    index*, so dropping tables means renumbering all of them; foreign keys whose
    other end was dropped are discarded. The result renders through
    `format_schema_prompt` exactly like an unpruned schema.
    """
    kept = sorted(keep_tables)
    table_map = {old: new for new, old in enumerate(kept)}

    column_map: dict[int, int] = {}
    column_names_original: list[list] = []
    column_types: list[str] = []
    for old_idx, (table_idx, col_name) in enumerate(schema["column_names_original"]):
        if table_idx != -1 and table_idx not in table_map:
            continue
        column_map[old_idx] = len(column_names_original)
        column_names_original.append([table_map.get(table_idx, -1), col_name])
        column_types.append(schema["column_types"][old_idx])

    return {
        "db_id": schema["db_id"],
        "table_names_original": [schema["table_names_original"][i] for i in kept],
        "column_names_original": column_names_original,
        "column_types": column_types,
        "primary_keys": [column_map[c] for c in schema["primary_keys"] if c in column_map],
        "foreign_keys": [
            [column_map[a], column_map[b]]
            for a, b in schema["foreign_keys"]
            if a in column_map and b in column_map
        ],
    }
