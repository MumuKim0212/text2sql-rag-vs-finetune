"""Condition 1 (cloud API baseline): generate SQL for the Spider dev set via a cloud LLM, then score it.

Usage:
    uv run python scripts/run_cloud_baseline.py --limit 20              # pilot run (Gemini, default)
    uv run python scripts/run_cloud_baseline.py --provider anthropic    # use Claude instead
    uv run python scripts/run_cloud_baseline.py                        # full dev set (1034 examples)

Predictions are appended to --out as JSONL, one line per example, so an
interrupted run can be resumed (already-written examples are skipped) without
re-paying for completed API calls.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from rag_text2sql.data import format_schema_prompt, load_schemas, load_spider
from rag_text2sql.eval import evaluate, summarize
from rag_text2sql.models import PROVIDERS

DB_DIR = Path("data/spider/test_suite_database")
TABLES_JSON = Path("data/spider/tables.json")
PLUG_VALUE = False  # official Spider default; recorded in summary.json so results stay self-describing


def _load_existing(out_path: Path) -> list[dict]:
    if not out_path.exists():
        return []
    with out_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N dev examples")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="gemini")
    parser.add_argument("--model", type=str, default=None, help="defaults to the provider's DEFAULT_MODEL")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    generate_sql, default_model = PROVIDERS[args.provider]
    model = args.model or default_model
    out_path = args.out or Path(f"data/results/cloud_baseline/{args.provider}_dev_predictions.jsonl")

    dev = load_spider()["dev"]
    if args.limit is not None:
        dev = dev.select(range(args.limit))
    schemas = load_schemas(TABLES_JSON)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing(out_path)
    start = len(done)
    if start:
        print(f"Resuming: {start} examples already done, skipping them.")

    with out_path.open("a", encoding="utf-8") as f:
        for i in range(start, len(dev)):
            example = dev[i]
            schema_prompt = format_schema_prompt(schemas[example["db_id"]])
            pred_sql = generate_sql(example["question"], schema_prompt, model=model)
            record = {
                "question": example["question"],
                "db_id": example["db_id"],
                "gold_sql": example["query"],
                "pred_sql": pred_sql,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(dev)}] {example['db_id']}: {pred_sql[:80]}")

    records = _load_existing(out_path)
    scores = evaluate(
        [r["pred_sql"] for r in records],
        [r["gold_sql"] for r in records],
        [r["db_id"] for r in records],
        DB_DIR,
        TABLES_JSON,
        plug_value=PLUG_VALUE,
    )
    result = {"provider": args.provider, "model": model, "plug_value": PLUG_VALUE, **summarize(scores)}
    print(result)
    (out_path.parent / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
