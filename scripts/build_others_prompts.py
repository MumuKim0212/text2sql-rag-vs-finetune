"""Build a second evaluation set from Spider's train_others split -- without value linking.

Motivation: value linking's effect is consistent in direction across five
measurements but has never cleared significance on Spider dev alone (n=1034).
With matcher v2's effect sizes it would take n~1550-1670, and dev has no more
examples -- test is private, and train_spider is what conditions 4/5 were
fine-tuned on. train_others is the one held-out pool left: disjoint from
train_spider by db_id (140 vs 6, no overlap), so neither the adapter nor the
retrieval index has seen it. The brief excludes it from the *training* and
*retrieval* sides only, which does not bar using it as extra evaluation.

**It cannot answer the value-linking question, though.** Spider ships these six
databases as schema only -- every table in all of them has zero rows -- so there
is no database content to ground a literal against. The only populated copies
are the test-suite instances, and those are generated from the gold queries:
sampling 40 academic questions, 39 of their gold literals ('Peter Mertens' and
the like) appear in the fuzzed data. Linking values from there would feed the
model strings derived from the gold SQL, which is exactly the leak the brief
forbids. So only the no-RAG prompts are built here.

What this set is still good for is any comparison that does not involve values:
conditions 2 -> 4 measure the fine-tuning effect, and running that on a
different query distribution tests whether the study's strongest result holds
outside Spider proper. The file below serves both conditions -- condition 2
replays it on the base model, condition 4 replays it with the adapter.

Only the 5 of 6 databases that have a test-suite database are kept, so every
prompt built here is scoreable on the same metric as dev (geo is dropped, 564
examples).

Usage:
    uv run python scripts/build_others_prompts.py
"""

import json
import statistics
from pathlib import Path

from rag_text2sql.data import format_schema_prompt, load_schemas
from rag_text2sql.rag import build_prompt

OTHERS_JSON = Path("data/spider/train_others.json")
TABLES_JSON = Path("data/spider/tables.json")
TEST_SUITE_DIR = Path("data/spider/test_suite_database")
OUT_PATH = Path("data/results/local_ft_others/prompts.jsonl")


def main() -> None:
    examples = json.loads(OTHERS_JSON.read_text(encoding="utf-8"))
    scoreable = {p.name for p in TEST_SUITE_DIR.iterdir() if p.is_dir()}
    dropped = sorted({e["db_id"] for e in examples} - scoreable)
    examples = [e for e in examples if e["db_id"] in scoreable]
    print(f"{len(examples)} examples kept; dropped databases without a test suite: {', '.join(dropped)}")

    schemas = load_schemas(TABLES_JSON)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lengths = []
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for example in examples:
            prompt = build_prompt(format_schema_prompt(schemas[example["db_id"]]), example["question"], [])
            lengths.append(len(prompt))
            record = {
                "question": example["question"],
                "db_id": example["db_id"],
                "gold_sql": example["query"],
                "prompt": prompt,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(examples)} prompts to {OUT_PATH}")
    print(f"  prompt chars: median {statistics.median(lengths):.0f}, max {max(lengths)}")
    print("  no value block: these databases are empty, see this file's docstring.")


if __name__ == "__main__":
    main()
