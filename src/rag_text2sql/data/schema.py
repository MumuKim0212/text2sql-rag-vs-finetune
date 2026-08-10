"""Render Spider tables.json schema entries as CREATE TABLE-style prompts."""

import json
from pathlib import Path


def load_schemas(tables_json: str | Path) -> dict[str, dict]:
    """Load tables.json, indexed by db_id."""
    with open(tables_json, encoding="utf-8") as f:
        tables = json.load(f)
    return {t["db_id"]: t for t in tables}


def format_schema_prompt(schema: dict) -> str:
    """Render one tables.json entry as CREATE TABLE statements (with PK/FK)."""
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]
    column_types = schema["column_types"]
    primary_keys = set(schema["primary_keys"])
    fk_by_col = dict(schema["foreign_keys"])

    columns_by_table: list[list[tuple[int, str, str]]] = [[] for _ in table_names]
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:
            continue
        columns_by_table[table_idx].append((col_idx, col_name, column_types[col_idx]))

    statements = []
    for table_idx, table_name in enumerate(table_names):
        lines = [
            f'  "{col_name}" {col_type}' + (" PRIMARY KEY" if col_idx in primary_keys else "")
            for col_idx, col_name, col_type in columns_by_table[table_idx]
        ]
        for col_idx, col_name, _ in columns_by_table[table_idx]:
            if col_idx in fk_by_col:
                ref_table_idx, ref_col_name = column_names[fk_by_col[col_idx]]
                ref_table_name = table_names[ref_table_idx]
                lines.append(f'  FOREIGN KEY ("{col_name}") REFERENCES "{ref_table_name}"("{ref_col_name}")')
        statements.append(f'CREATE TABLE "{table_name}" (\n' + ",\n".join(lines) + "\n)")

    return "\n\n".join(statements)
