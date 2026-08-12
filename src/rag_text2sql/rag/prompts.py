"""Assemble the RAG user prompt from retrieved examples plus the (possibly pruned) schema."""


def build_prompt(
    schema_prompt: str,
    question: str,
    examples: list[tuple[str, str]],
    values: list[tuple[str, str, str]] | None = None,
) -> str:
    """Prepend retrieved (question, SQL) demonstrations to condition 2's prompt.

    Without `values`, everything from "Schema:" onwards is byte-identical to the
    no-RAG prompt, so the retrieved examples are the only difference between the
    two conditions. `values` adds linked database literals just before the
    question, where they sit closest to the text they need to correct.
    """
    demonstrations = "\n\n".join(f"Question: {q}\nSQL: {sql}" for q, sql in examples)
    value_block = ""
    if values:
        linked = "\n".join(f'  {table}.{column} = "{value}"' for table, column, value in values)
        value_block = f"Values in the database matching words in the question:\n{linked}\n\n"
    return (
        f"Here are examples of questions and the SQL that answers them:\n\n{demonstrations}\n\n"
        f"Schema:\n{schema_prompt}\n\n"
        f"{value_block}"
        f"Question: {question}\nSQL:"
    )
