"""Assemble the RAG user prompt from retrieved examples plus the (possibly pruned) schema."""


def build_prompt(
    schema_prompt: str,
    question: str,
    examples: list[tuple[str, str]],
    values: list[tuple[str, str, str]] | None = None,
) -> str:
    """Add retrieved demonstrations and/or linked database values to condition 2's prompt.

    Both blocks are optional and independent, which is what makes conditions
    2/3a/3d/3c a clean 2x2 over (few-shot, values): with neither, this returns
    condition 2's prompt exactly, so each block's contribution is separable.
    Values sit just before the question, closest to the text they correct.
    """
    demonstrations = ""
    if examples:
        rendered = "\n\n".join(f"Question: {q}\nSQL: {sql}" for q, sql in examples)
        demonstrations = f"Here are examples of questions and the SQL that answers them:\n\n{rendered}\n\n"
    value_block = ""
    if values:
        linked = "\n".join(f'  {table}.{column} = "{value}"' for table, column, value in values)
        value_block = f"Values in the database matching words in the question:\n{linked}\n\n"
    return (
        f"{demonstrations}"
        f"Schema:\n{schema_prompt}\n\n"
        f"{value_block}"
        f"Question: {question}\nSQL:"
    )
