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


def _to_single_line(sql: str) -> str:
    """Flatten one query to a single non-empty line for the evaluator's file format.

    The vendored evaluator parses the gold and prediction files line-by-line:
    a newline inside one query silently shifts every later prediction onto the
    wrong gold (its zip() just truncates, no error), and a query that is empty
    after stripping is read as a session separator and trips an assert. Local
    models routinely emit multi-line SQL, so both are normalized away here.
    Empty predictions become a query that fails to execute, i.e. scored wrong.
    """
    return " ".join(sql.splitlines()).strip() or "SELECT"


def evaluate(
    predicted_sql: list[str],
    gold_sql: list[str],
    gold_db_ids: list[str],
    db_dir: str | Path,
    tables_json: str | Path,
    etype: str = "all",
    plug_value: bool = False,
) -> dict:
    """Score predicted SQL against gold SQL using Spider's official metrics.

    `db_dir` must be the *test-suite* database directory (multiple fuzzed
    sqlite instances per db_id) -- pointing this at the standard
    single-instance `database/` folder silently degenerates Test-Suite
    Accuracy into plain single-instance execution accuracy.

    `plug_value=False` (default) matches the official Spider evaluation's own
    default, which is what leaderboard numbers are measured with. Setting it
    True additionally scores variants of the prediction with the gold query's
    literal values plugged in, crediting structure/schema linking even when a
    literal is wrong -- a strictly easier setting whose scores must not be
    compared against the leaderboard.

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
            "\n".join(
                f"{_to_single_line(sql)}\t{db_id}" for sql, db_id in zip(gold_sql, gold_db_ids)
            ),
            encoding="utf-8",
        )
        pred_path.write_text("\n".join(_to_single_line(sql) for sql in predicted_sql), encoding="utf-8")

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


DIFFICULTY_LEVELS = ("easy", "medium", "hard", "extra")


def summarize(scores: dict, etype: str = "all") -> dict:
    """Pull the headline "all"-level numbers used in the 5-condition comparison table.

    `etype` must match what was passed to `evaluate()`: the scores dict always
    carries an 'exact' key, but it's only meaningfully computed when etype was
    "all" or "match" -- otherwise it's reported as None here rather than a
    misleading 0.0.

    The per-difficulty split is computed by the evaluator anyway, so it's kept
    here too: an aggregate that moves says a condition helped, but not whether
    it helped on the simple queries or the ones that were actually hard.
    """
    scored_exact = etype in ("all", "match")

    def _level(level: str) -> dict:
        return {
            "test_suite_accuracy": scores[level]["exec"],
            "exact_match": scores[level]["exact"] if scored_exact else None,
            "n": scores[level]["count"],
        }

    return {
        **_level("all"),
        "by_difficulty": {level: _level(level) for level in DIFFICULTY_LEVELS},
    }
