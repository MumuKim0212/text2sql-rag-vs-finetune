"""Condition 3: build the RAG prompts for the Spider dev set, ahead of generation.

Retrieval, schema linking and value linking run here (CPU, on the machine
holding the Spider data); the GPU step only replays the finished prompts. All
three variants are built in one pass so they share the exact same retrieved
examples, leaving one deliberate difference each: 3b prunes the schema, 3c adds
linked database values.

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
    load_db_values,
    load_embedder,
    match_values,
    prune_schema,
    table_texts,
    top_k,
)

TABLES_JSON = Path("data/spider/tables.json")
DB_DIR = Path("data/spider/database")  # canonical Spider DBs -- what a deployment would query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k-examples", type=int, default=5, help="few-shot examples per prompt")
    # k=5 over k=4: same pruning coverage (252 vs 256 dev examples), but it drops a
    # table the gold query needs in 6 cases instead of 22.
    parser.add_argument("--top-k-tables", type=int, default=5, help="tables kept by schema linking")
    parser.add_argument("--out-dir", type=Path, default=Path("data/results"))
    # A measured condition's prompts are the record of what was actually run, so a
    # rebuild under changed settings goes to its own directories rather than over them.
    parser.add_argument("--tag", default="", help="suffix for the output directories, e.g. _v2")
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

    print(f"Reading cell values from {len(set(dev_db_ids))} dev databases...")
    db_values = {db: load_db_values(DB_DIR / db / f"{db}.sqlite") for db in sorted(set(dev_db_ids))}

    paths = {
        "few-shot": args.out_dir / f"local_rag{args.tag}" / "prompts.jsonl",
        "+ schema linking": args.out_dir / f"local_rag_linked{args.tag}" / "prompts.jsonl",
        "+ value linking": args.out_dir / f"local_rag_values{args.tag}" / "prompts.jsonl",
        "values only": args.out_dir / f"local_rag_values_only{args.tag}" / "prompts.jsonl",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    gold_sql = dev["query"]
    n_pruned = n_valued = n_hits = 0
    lengths = {name: [] for name in paths}

    files = {name: path.open("w", encoding="utf-8") for name, path in paths.items()}
    try:
        for i, question in enumerate(dev_questions):
            db_id = dev_db_ids[i]
            schema = schemas[db_id]
            examples = [(train_questions[j], train_queries[j]) for j in neighbours[i]]

            keep = top_k(dev_embeddings[i : i + 1], table_embeddings[db_id], args.top_k_tables)[0].tolist()
            n_pruned += len(keep) < len(schema["table_names_original"])

            hits = match_values(question, db_values[db_id])
            n_valued += bool(hits)
            n_hits += len(hits)

            record = {"question": question, "db_id": db_id, "gold_sql": gold_sql[i]}
            # (variant, schema, examples, values) -- "values only" drops the demonstrations
            # so that conditions 2/3a/3d/3c form a 2x2 over (few-shot, values).
            variants = (
                ("few-shot", format_schema_prompt(schema), examples, None),
                ("+ schema linking", format_schema_prompt(prune_schema(schema, keep)), examples, None),
                ("+ value linking", format_schema_prompt(schema), examples, hits),
                ("values only", format_schema_prompt(schema), [], hits),
            )
            for name, schema_prompt, shots, values in variants:
                prompt = build_prompt(schema_prompt, question, shots, values)
                lengths[name].append(len(prompt))
                files[name].write(json.dumps({**record, "prompt": prompt}, ensure_ascii=False) + "\n")
    finally:
        for f in files.values():
            f.close()

    n = len(dev_questions)
    print(f"\nWrote {n} prompts to each of: {', '.join(str(p) for p in paths.values())}")
    print(f"Schema linking changed {n_pruned}/{n} ({n_pruned / n:.1%}) prompts "
          f"-- the rest are in databases with <= {args.top_k_tables} tables, where it is a no-op.")
    print(f"Value linking attached at least one value to {n_valued}/{n} ({n_valued / n:.1%}) prompts, "
          f"{n_hits / n:.1f} values per prompt on average.")
    for name, values in lengths.items():
        print(f"  {name:16s} prompt chars: median {statistics.median(values):.0f}, max {max(values)}")


if __name__ == "__main__":
    main()
