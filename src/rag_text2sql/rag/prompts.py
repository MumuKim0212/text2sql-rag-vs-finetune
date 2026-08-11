"""Assemble the RAG user prompt from retrieved examples plus the (possibly pruned) schema."""


def build_prompt(schema_prompt: str, question: str, examples: list[tuple[str, str]]) -> str:
    """Prepend retrieved (question, SQL) demonstrations to condition 2's prompt.

    Everything from "Schema:" onwards is byte-identical to the no-RAG prompt, so
    the retrieved examples are the only difference between the two conditions.
    """
    demonstrations = "\n\n".join(f"Question: {q}\nSQL: {sql}" for q, sql in examples)
    return (
        f"Here are examples of questions and the SQL that answers them:\n\n{demonstrations}\n\n"
        f"Schema:\n{schema_prompt}\n\nQuestion: {question}\nSQL:"
    )
