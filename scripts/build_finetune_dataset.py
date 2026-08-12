"""Condition 4: export the Spider train split as chat-format SFT data.

Runs locally so the Colab notebook only has to train, and so the exact training
set is a tracked artifact next to the predictions it produces.

Usage:
    uv run python scripts/build_finetune_dataset.py
"""

import argparse
import json
import statistics
from pathlib import Path

from rag_text2sql.data import load_schemas, load_spider, unique_db_ids
from rag_text2sql.finetune.dataset import build_messages

TABLES_JSON = Path("data/spider/tables.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/results/local_ft/train_messages.jsonl"))
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


if __name__ == "__main__":
    main()
