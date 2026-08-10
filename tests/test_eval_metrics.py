from pathlib import Path

from rag_text2sql.data import load_spider
from rag_text2sql.eval import evaluate, summarize

DB_DIR = Path("data/spider/test_suite_database")
TABLES_JSON = Path("data/spider/tables.json")

N = 8


def _dev_sample():
    dev = load_spider()["dev"]
    return dev["query"][:N], dev["db_id"][:N]


def test_perfect_predictions_score_1():
    gold_sql, gold_db_ids = _dev_sample()

    scores = evaluate(list(gold_sql), gold_sql, gold_db_ids, DB_DIR, TABLES_JSON)
    result = summarize(scores)

    assert result["n"] == N
    assert result["test_suite_accuracy"] == 1.0
    assert result["exact_match"] == 1.0


def test_wrong_predictions_score_0():
    gold_sql, gold_db_ids = _dev_sample()
    wrong_sql = ["SELECT 1"] * N

    scores = evaluate(wrong_sql, gold_sql, gold_db_ids, DB_DIR, TABLES_JSON, etype="exec")
    result = summarize(scores, etype="exec")

    assert result["test_suite_accuracy"] == 0.0
    assert result["exact_match"] is None
