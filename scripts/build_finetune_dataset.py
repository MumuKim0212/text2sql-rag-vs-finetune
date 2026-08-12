"""Condition 4: export the Spider train split as chat-format SFT data, plus its dev prompts.

Runs locally so the Colab notebook only has to train, and so the exact training
set is a tracked artifact next to the predictions it produces.

The dev prompts are condition 2's prompt exactly -- no retrieved examples, no
linked values -- since condition 4 is fine-tuning without RAG. Condition 2 built
these inline in its notebook and never saved them, so this also puts that input
on disk. Condition 5 does not need a new file: it reuses 3d's prompts
(`data/results/local_rag_values_only/prompts.jsonl`).

Usage:
    uv run python scripts/build_finetune_dataset.py
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
from rag_text2sql.finetune.dataset import build_messages
from rag_text2sql.rag import build_prompt

TABLES_JSON = Path("data/spider/tables.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/results/local_ft/train_messages.jsonl"))
    parser.add_argument(
        "--dev-prompts-out", type=Path, default=Path("data/results/local_ft/prompts.jsonl")
    )
    args = parser.parse_args()

    spider = load_spider()
    train, dev = spider["train"], spider["dev"]
    if not unique_db_ids(train).isdisjoint(unique_db_ids(dev)):
        raise ValueError("train/dev db_id overlap -- fine-tuning would train on the evaluation set")

    schemas = load_schemas(TABLES_JSON)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    questions, db_ids, queries = train["question"], train["db_id"], train["query"]
    lengths = []
    with args.out.open("w", encoding="utf-8") as f:
        for question, db_id, gold_sql in zip(questions, db_ids, queries):
            messages = build_messages(question, schemas[db_id], gold_sql)
            lengths.append(sum(len(m["content"]) for m in messages))
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(lengths)} examples to {args.out}")
    print(f"  chars per example: median {statistics.median(lengths):.0f}, max {max(lengths)}")
    print(f"  databases covered: {len(unique_db_ids(train))} (dev's 20 are disjoint from these)")

    n_dev = 0
    with args.dev_prompts_out.open("w", encoding="utf-8") as f:
        for question, db_id, gold_sql in zip(dev["question"], dev["db_id"], dev["query"]):
            prompt = build_prompt(format_schema_prompt(schemas[db_id]), question, [])
            record = {"question": question, "db_id": db_id, "gold_sql": gold_sql, "prompt": prompt}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_dev += 1

    print(f"Wrote {n_dev} dev prompts to {args.dev_prompts_out} (condition 2's prompt, no RAG)")


if __name__ == "__main__":
    main()
