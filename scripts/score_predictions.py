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
PLUG_VALUE = False  # official Spider default; recorded in summary.json so results stay self-describing


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
        plug_value=PLUG_VALUE,
    )
    out_dir = Path(f"data/results/{args.condition}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    result = {"condition": args.condition, "plug_value": PLUG_VALUE, **summarize(scores)}
    print(result)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Per-example results, so paired comparisons between conditions can be run
    # without re-executing every query against the test-suite databases.
    per_example = scores["per_example"]
    if len(per_example) != len(records):
        raise RuntimeError(f"per-example results ({len(per_example)}) do not line up with predictions ({len(records)})")
    with (out_dir / "per_example.jsonl").open("w", encoding="utf-8") as f:
        for i, (record, entry) in enumerate(zip(records, per_example)):
            f.write(
                json.dumps(
                    {
                        "i": i,
                        "db_id": record["db_id"],
                        "hardness": entry["hardness"],
                        "exec": int(entry["exec"]),
                        "exact": int(entry["exact"]),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
