"""Condition 3: build the RAG prompts for the Spider dev set, ahead of generation.

Retrieval and schema linking run here (CPU, on the machine holding the Spider
data); the GPU step only replays the finished prompts. Both variants are built
in one pass so they share the exact same retrieved examples -- the pruned schema
is then the only difference between them.

Usage:
    uv run python scripts/build_rag_prompts.py
"""

import argparse
import json
import statistics
from pathlib import Path

from rag_text2sql.data import (
    format_schema_prompt,
    load_schemas,
    load_spider,
    unique_db_ids,
)
from rag_text2sql.rag import (
    build_prompt,
    encode,
    load_embedder,
    prune_schema,
    table_texts,
    top_k,
)

TABLES_JSON = Path("data/spider/tables.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k-examples", type=int, default=5, help="few-shot examples per prompt")
    # k=5 over k=4: same pruning coverage (252 vs 256 dev examples), but it drops a
    # table the gold query needs in 6 cases instead of 22.
    parser.add_argument("--top-k-tables", type=int, default=5, help="tables kept by schema linking")
    parser.add_argument("--out-dir", type=Path, default=Path("data/results"))
    args = parser.parse_args()

    spider = load_spider()
    train, dev = spider["train"], spider["dev"]
    if not unique_db_ids(train).isdisjoint(unique_db_ids(dev)):
        raise ValueError("train/dev db_id overlap -- retrieval would leak the evaluation set")

    embedder = load_embedder()
    train_questions, train_queries = train["question"], train["query"]
    print(f"Embedding {len(train_questions)} train questions (retrieval index)...")
    index = encode(embedder, train_questions)

    dev_questions = dev["question"]
    print(f"Embedding {len(dev_questions)} dev questions...")
    dev_embeddings = encode(embedder, dev_questions)
    neighbours = top_k(dev_embeddings, index, args.top_k_examples)

    schemas = load_schemas(TABLES_JSON)
    dev_db_ids = dev["db_id"]
    table_embeddings = {db: encode(embedder, table_texts(schemas[db])) for db in sorted(set(dev_db_ids))}

    plain_path = args.out_dir / "local_rag" / "prompts.jsonl"
    linked_path = args.out_dir / "local_rag_linked" / "prompts.jsonl"
    for path in (plain_path, linked_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    gold_sql = dev["query"]
    n_pruned = 0
    plain_lengths, linked_lengths = [], []

    with plain_path.open("w", encoding="utf-8") as plain_f, linked_path.open("w", encoding="utf-8") as linked_f:
        for i, question in enumerate(dev_questions):
            db_id = dev_db_ids[i]
            schema = schemas[db_id]
            examples = [(train_questions[j], train_queries[j]) for j in neighbours[i]]

            keep = top_k(dev_embeddings[i : i + 1], table_embeddings[db_id], args.top_k_tables)[0].tolist()
            n_pruned += len(keep) < len(schema["table_names_original"])

            record = {"question": question, "db_id": db_id, "gold_sql": gold_sql[i]}
            for f, schema_prompt, lengths in (
                (plain_f, format_schema_prompt(schema), plain_lengths),
                (linked_f, format_schema_prompt(prune_schema(schema, keep)), linked_lengths),
            ):
                prompt = build_prompt(schema_prompt, question, examples)
                lengths.append(len(prompt))
                f.write(json.dumps({**record, "prompt": prompt}, ensure_ascii=False) + "\n")

    n = len(dev_questions)
    print(f"\nWrote {n} prompts to {plain_path} and {linked_path}")
    print(f"Schema linking changed {n_pruned}/{n} ({n_pruned / n:.1%}) prompts "
          f"-- the rest are in databases with <= {args.top_k_tables} tables, where it is a no-op.")
    for name, lengths in (("few-shot", plain_lengths), ("+ schema linking", linked_lengths)):
        print(f"  {name:16s} prompt chars: median {statistics.median(lengths):.0f}, max {max(lengths)}")


if __name__ == "__main__":
    main()
