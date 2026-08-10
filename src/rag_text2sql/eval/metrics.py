"""Test-Suite Accuracy (+ Exact Match) evaluation, reused across all 5 experiment conditions.

Wraps the vendored official Spider evaluation code (see
third_party/test_suite_sql_eval/) so scores can be consumed as a dict instead
of parsed from printed text.
"""

import sys
import tempfile
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parents[3] / "third_party" / "test_suite_sql_eval"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

import nltk
from evaluation import build_foreign_key_map_from_json
from evaluation import evaluate as _ts_evaluate

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def evaluate(
    predicted_sql: list[str],
    gold_sql: list[str],
    gold_db_ids: list[str],
    db_dir: str | Path,
    tables_json: str | Path,
    etype: str = "all",
    plug_value: bool = True,
) -> dict:
    """Score predicted SQL against gold SQL using Spider's official metrics.

    `db_dir` must be the *test-suite* database directory (multiple fuzzed
    sqlite instances per db_id) -- pointing this at the standard
    single-instance `database/` folder silently degenerates Test-Suite
    Accuracy into plain single-instance execution accuracy.

    `plug_value=True` (default) substitutes the gold query's literal values
    into the predicted query before execution, matching the brief's Text-to-
    SQL setup where models are scored on structure/schema linking rather than
    exact literal reproduction.

    Returns the vendored evaluate()'s scores dict, keyed by difficulty level
    ("easy"/"medium"/"hard"/"extra"/"all"/"joint_all"), each with 'exec'
    (Test-Suite Accuracy) and, when etype is "all" or "match", 'exact'
    (Exact Match).
    """
    if not (len(predicted_sql) == len(gold_sql) == len(gold_db_ids)):
        raise ValueError("predicted_sql, gold_sql, and gold_db_ids must be the same length")

    kmaps = build_foreign_key_map_from_json(str(tables_json)) if etype in ("all", "match") else None

    with tempfile.TemporaryDirectory() as tmp:
        gold_path = Path(tmp) / "gold.txt"
        pred_path = Path(tmp) / "pred.txt"
        gold_path.write_text(
            "\n".join(f"{sql}\t{db_id}" for sql, db_id in zip(gold_sql, gold_db_ids)),
            encoding="utf-8",
        )
        pred_path.write_text("\n".join(predicted_sql), encoding="utf-8")

        return _ts_evaluate(
            str(gold_path),
            str(pred_path),
            str(db_dir),
            etype,
            kmaps,
            plug_value,
            False,  # keep_distinct
            False,  # progress_bar_for_each_datapoint
        )


def summarize(scores: dict, etype: str = "all") -> dict:
    """Pull the headline "all"-level numbers used in the 5-condition comparison table.

    `etype` must match what was passed to `evaluate()`: the scores dict always
    carries an 'exact' key, but it's only meaningfully computed when etype was
    "all" or "match" -- otherwise it's reported as None here rather than a
    misleading 0.0.
    """
    all_scores = scores["all"]
    return {
        "test_suite_accuracy": all_scores["exec"],
        "exact_match": all_scores["exact"] if etype in ("all", "match") else None,
        "n": all_scores["count"],
    }
