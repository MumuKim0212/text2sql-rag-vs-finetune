"""Score an existing predictions.jsonl file against Spider dev using test-suite accuracy.

Generation for local-model conditions (2-5) happens elsewhere (e.g. a Colab
notebook, since this machine has no usable GPU) and only produces a
predictions.jsonl with the same {question, db_id, gold_sql, pred_sql} schema
as scripts/run_cloud_baseline.py. Scoring needs the test-suite database
(4.9GB, local-only), so it's kept as a separate step reused across conditions.

Usage:
    uv run python scripts/score_predictions.py --predictions data/results/local_base/qwen_dev_predictions.jsonl --condition local_base
"""

import argparse
import json
from pathlib import Path

from rag_text2sql.eval import evaluate, summarize

DB_DIR = Path("data/spider/test_suite_database")
TABLES_JSON = Path("data/spider/tables.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--condition", type=str, required=True, help="e.g. local_base, local_rag, local_ft")
    args = parser.parse_args()

    with args.predictions.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    scores = evaluate(
        [r["pred_sql"] for r in records],
        [r["gold_sql"] for r in records],
        [r["db_id"] for r in records],
        DB_DIR,
        TABLES_JSON,
    )
    result = {"condition": args.condition, **summarize(scores)}
    print(result)

    out_dir = Path(f"data/results/{args.condition}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
