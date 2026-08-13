"""Paired significance tests, difficulty breakdown, and an error taxonomy for the gap to the cloud baseline.

Reads the `per_example.jsonl` files written by scripts/score_predictions.py, so
it needs no GPU and never re-executes a query against the test-suite databases.

The comparison table in RESULTS.md reports differences of a few examples out of
1034 (value injection: +7 and +8). Those were defended against an empirically
measured decoding-noise floor, which says nothing about sampling variance --
this computes the paired test the data supports instead (McNemar exact, since
every condition is scored on the same 1034 examples).

Usage:
    uv run python scripts/analyze_results.py
"""

import json
import sys
from math import comb, exp, factorial, log
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parents[1] / "third_party" / "test_suite_sql_eval"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from process_sql import Schema, get_schema, get_sql

RESULTS_DIR = Path("data/results")
DB_DIR = Path("data/spider/test_suite_database")

CONDITIONS = {
    "cloud_baseline": "1. 클라우드 API",
    "local_base": "2. 로컬 베이스",
    "local_rag": "3a. + few-shot",
    "local_rag_linked": "3b. + schema linking",
    "local_rag_values": "3c. + few-shot + 값",
    "local_rag_values_only": "3d. + 값 주입",
    "local_ft": "4. + 파인튜닝",
    "local_ft_rag": "5. + 파인튜닝 + 값",
    "local_rag_values_only_v2": "3d′. + 값 주입 (매처 v2)",
    "local_ft_rag_v2": "5′. + 파인튜닝 + 값 (매처 v2)",
    "local_base_others": "2. 로컬 베이스 (train_others)",
    "local_ft_others": "4. + 파인튜닝 (train_others)",
}

# Conditions are only ever compared within an evaluation set: dev has 1034 examples
# and train_others 1095, and every comparison here is paired example-by-example.
EVAL_SETS = {
    "Spider dev (n=1034)": [c for c in CONDITIONS if not c.endswith("_others")],
    "train_others (n=1095)": [c for c in CONDITIONS if c.endswith("_others")],
}

# (baseline, treatment, what the treatment adds)
PAIRS = [
    ("local_base", "local_rag", "few-shot 예제 검색"),
    ("local_rag", "local_rag_linked", "schema linking"),
    ("local_rag", "local_rag_values", "값 주입 (few-shot 위)"),
    ("local_base", "local_rag_values_only", "값 주입 (단독)"),
    ("local_ft", "local_ft_rag", "값 주입 (파인튜닝 위)"),
    ("local_base", "local_ft", "QLoRA 파인튜닝"),
    ("local_rag_values_only", "local_ft_rag", "QLoRA 파인튜닝 (값 위)"),
    ("local_ft_rag", "cloud_baseline", "로컬 최고 → 클라우드"),
    ("local_base", "local_rag_values_only_v2", "값 주입 v2 (단독)"),
    ("local_ft", "local_ft_rag_v2", "값 주입 v2 (파인튜닝 위)"),
    ("local_rag_values_only", "local_rag_values_only_v2", "매처 v1 → v2 (단독)"),
    ("local_ft_rag", "local_ft_rag_v2", "매처 v1 → v2 (파인튜닝 위)"),
    ("local_base_others", "local_ft_others", "QLoRA 파인튜닝 (train_others)"),
]

# Each set holds the two independent measurements of one matcher's value-injection
# effect -- with and without fine-tuning -- which Fisher's method combines. v1 and
# v2 are NOT independent of each other (same 1034 examples), so they never combine
# across sets; each is reported on its own.
VALUE_REPLICATIONS = {
    "매처 v1": [("local_base", "local_rag_values_only"), ("local_ft", "local_ft_rag")],
    "매처 v2": [("local_base", "local_rag_values_only_v2"), ("local_ft", "local_ft_rag_v2")],
}

DIFFICULTIES = ("easy", "medium", "hard", "extra")

# Join syntax Spider's own queries never use, so its parser rejects it as an unknown table.
JOIN_KEYWORDS = {"inner", "left", "right", "outer", "full", "cross", "natural"}


def mcnemar_exact(a: list[int], b: list[int]) -> dict:
    """Two-sided exact McNemar test on paired binary outcomes.

    Only examples the two conditions disagree on carry information: under the
    null that the treatment changes nothing, each disagreement is a coin flip,
    so the number won by one side is Binomial(discordant, 0.5).
    """
    b_only = sum(1 for x, y in zip(a, b) if y and not x)  # treatment right, baseline wrong
    a_only = sum(1 for x, y in zip(a, b) if x and not y)  # baseline right, treatment wrong
    n = b_only + a_only
    if n == 0:
        return {"gained": 0, "lost": 0, "net": 0, "p": 1.0}
    lo = min(b_only, a_only)
    tail = sum(comb(n, k) for k in range(lo + 1)) / 2**n
    return {"gained": b_only, "lost": a_only, "net": b_only - a_only, "p": min(1.0, 2 * tail)}


def fisher_combined(p_values: list[float]) -> float:
    """Fisher's method: combine independent p-values into one.

    chi2 = -2*sum(ln p) on 2k degrees of freedom. Only even df arise here, so
    the survival function is the closed form exp(-x/2) * sum (x/2)^i / i!.
    """
    x = -2 * sum(log(p) for p in p_values)
    k = len(p_values)  # df = 2k
    return exp(-x / 2) * sum((x / 2) ** i / factorial(i) for i in range(k))


def load_per_example(condition: str) -> list[dict]:
    path = RESULTS_DIR / condition / "per_example.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_predictions(condition: str) -> list[dict]:
    (path,) = sorted(RESULTS_DIR.joinpath(condition).glob("*_predictions.jsonl"))
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt_p(p: float) -> str:
    return f"{p:.3g}" if p >= 1e-4 else "<1e-4"


def available_pairs(per_example: dict[str, list[dict]]) -> list[tuple[str, str, str]]:
    """The comparisons both of whose conditions have been scored."""
    return [(b, t, label) for b, t, label in PAIRS if b in per_example and t in per_example]


def print_difficulty_table(per_example: dict[str, list[dict]]) -> None:
    print("\n## 난이도별 Test-Suite Accuracy (정답 수 / 문항 수)\n")
    any_rows = next(iter(per_example.values()))
    counts = {d: sum(1 for r in any_rows if r["hardness"] == d) for d in DIFFICULTIES}
    print("| 조건 | " + " | ".join(f"{d} (n={counts[d]})" for d in DIFFICULTIES) + " | all |")
    print("|---" * (len(DIFFICULTIES) + 2) + "|")
    for cond, rows in per_example.items():
        label = CONDITIONS[cond]
        cells = []
        for d in DIFFICULTIES:
            hit = sum(r["exec"] for r in rows if r["hardness"] == d)
            cells.append(f"{hit} ({hit / counts[d]:.1%})")
        total = sum(r["exec"] for r in rows)
        print(f"| {label} | " + " | ".join(cells) + f" | {total} ({total / len(rows):.1%}) |")


def print_difficulty_deltas(per_example: dict[str, list[dict]]) -> None:
    print("\n## 기법별 난이도 구간 기여 (Test-Suite Accuracy 정답 수 변화)\n")
    print("| 추가한 것 | " + " | ".join(DIFFICULTIES) + " | 합계 |")
    print("|---" * (len(DIFFICULTIES) + 2) + "|")
    for base, treat, label in available_pairs(per_example):
        cells = []
        for d in DIFFICULTIES:
            delta = sum(r["exec"] for r in per_example[treat] if r["hardness"] == d) - sum(
                r["exec"] for r in per_example[base] if r["hardness"] == d
            )
            cells.append(f"{delta:+d}")
        total = sum(r["exec"] for r in per_example[treat]) - sum(r["exec"] for r in per_example[base])
        print(f"| {label} | " + " | ".join(cells) + f" | {total:+d} |")


def print_mcnemar(per_example: dict[str, list[dict]]) -> None:
    print("\n## McNemar 정확검정 (양측)\n")
    print("| 추가한 것 | 지표 | 얻음 | 잃음 | 순변화 | p |")
    print("|---|---|---|---|---|---|")
    for base, treat, label in available_pairs(per_example):
        for metric in ("exec", "exact"):
            r = mcnemar_exact(
                [x[metric] for x in per_example[base]],
                [x[metric] for x in per_example[treat]],
            )
            name = "실행" if metric == "exec" else "EM"
            print(f"| {label} | {name} | {r['gained']} | {r['lost']} | {r['net']:+d} | {fmt_p(r['p'])} |")

    for name, replications in VALUE_REPLICATIONS.items():
        if not all(c in per_example for pair in replications for c in pair):
            continue
        p_values = [
            mcnemar_exact([x["exec"] for x in per_example[b]], [x["exec"] for x in per_example[t]])["p"]
            for b, t in replications
        ]
        print(
            f"\n{name}: 값 주입 두 독립 측정({' , '.join(f'p={fmt_p(p)}' for p in p_values)})의 "
            f"Fisher 결합: **p = {fmt_p(fisher_combined(p_values))}**"
        )


def _strip_conds(clause: list) -> list:
    """Blank out the literal operands of a WHERE/HAVING clause, keeping columns and operators.

    A cond_unit is (not_op, op_id, val_unit, val1, val2); only val1/val2 hold
    literals, and either can instead be a nested query, which is normalized too.
    """
    out = []
    for cond in clause:
        if cond in ("and", "or"):
            out.append(cond)
            continue
        not_op, op_id, val_unit, val1, val2 = cond
        out.append((not_op, op_id, val_unit, *(_norm(v) if isinstance(v, dict) else "?" for v in (val1, val2))))
    return out


def _norm(sql: dict) -> dict:
    """A copy of a parsed query with literal values removed, so structure compares on its own."""
    return {
        "select": sql["select"],
        "from": {
            "table_units": [(t, _norm(u) if t == "sql" else u) for t, u in sql["from"]["table_units"]],
            "conds": _strip_conds(sql["from"]["conds"]),
        },
        "where": _strip_conds(sql["where"]),
        "groupBy": sql["groupBy"],
        "having": _strip_conds(sql["having"]),
        "orderBy": sql["orderBy"],
        "limit": sql["limit"],
        **{key: _norm(sql[key]) if sql[key] else None for key in ("intersect", "union", "except")},
    }


def _tables(parsed: dict) -> set:
    return {u[1] for u in parsed["from"]["table_units"] if u[0] == "table_unit"}


def _is_nested(parsed: dict) -> bool:
    if any(parsed[key] for key in ("intersect", "union", "except")):
        return True
    if any(t == "sql" for t, _ in parsed["from"]["table_units"]):
        return True
    for clause in (parsed["where"], parsed["having"]):
        for cond in clause:
            if cond not in ("and", "or") and any(isinstance(v, dict) for v in cond[3:5]):
                return True
    return False


def classify(pred: str, gold: str, db_id: str, schema_cache: dict) -> str:
    """Bucket one failure by its first structural mismatch against the gold query."""
    if db_id not in schema_cache:
        schema_cache[db_id] = Schema(get_schema(str(DB_DIR / db_id / f"{db_id}.sqlite")))
    schema = schema_cache[db_id]
    try:
        g = get_sql(schema, gold)
    except Exception:  # noqa: BLE001 -- the vendored parser raises whatever it hits
        return "gold 파싱 실패"
    try:
        p = get_sql(schema, pred)
    except KeyError as e:
        # Table lookup: the parser resolves table names through the schema, so an
        # unsupported join keyword and a table that doesn't exist raise the same
        # KeyError. Only the latter is a model error.
        token = str(e.args[0]).strip("'\" ").lower() if e.args else ""
        return "공식 파서 미지원 문법" if token in JOIN_KEYWORDS else "스키마에 없는 컬럼·테이블 참조"
    except AssertionError as e:
        # Column lookup fails with this message; every other assertion is the
        # parser refusing a construct Spider's own queries never use.
        return "스키마에 없는 컬럼·테이블 참조" if str(e).startswith("Error col:") else "공식 파서 미지원 문법"
    except Exception:  # noqa: BLE001 -- same, for constructs neither branch above named
        return "공식 파서 미지원 문법"

    np_, ng = _norm(p), _norm(g)
    if _tables(p) != _tables(g):
        return "테이블 선택(조인 대상)"
    if _is_nested(p) != _is_nested(g):
        return "중첩·집합연산 유무"
    if np_["groupBy"] != ng["groupBy"] or np_["having"] != ng["having"]:
        return "GROUP BY / HAVING"
    if np_["select"] != ng["select"]:
        return "SELECT 절(컬럼·집계)"
    if np_["where"] != ng["where"]:
        return "WHERE 조건(컬럼·연산자)"
    if (p["where"], p["having"]) != (g["where"], g["having"]):
        return "리터럴 값만 불일치"
    if np_["orderBy"] != ng["orderBy"] or np_["limit"] != ng["limit"]:
        return "ORDER BY / LIMIT"
    return "구조 동일(실행 결과만 불일치)"


def print_parse_failures(per_example: dict[str, list[dict]]) -> None:
    """How often the official parser rejects a prediction, and what that costs.

    Spider's parser only accepts the subset of SQL its own queries are written
    in: plain `JOIN` but not `INNER`/`LEFT JOIN`, single-column `GROUP BY`, no
    `AS` on a select expression. The evaluator swaps an unparseable prediction
    for an empty query, so Exact Match is forced to 0 -- even for a prediction
    that executed correctly. Execution accuracy runs the raw string and is
    unaffected, which makes this a pure Exact Match artifact worth sizing.
    """
    print("\n## 공식 파서 파싱 실패율 (Exact Match에만 영향)\n")
    print("| 조건 | 파싱 실패 | 그중 실행은 정답 | 강제로 잃은 EM |")
    print("|---|---|---|---|")
    schema_cache: dict = {}
    unparseable: dict[str, set[int]] = {}
    for cond, rows in per_example.items():
        label = CONDITIONS[cond]
        records = load_predictions(cond)
        failed = set()
        for i, r in enumerate(records):
            db_id = r["db_id"]
            if db_id not in schema_cache:
                schema_cache[db_id] = Schema(get_schema(str(DB_DIR / db_id / f"{db_id}.sqlite")))
            try:
                get_sql(schema_cache[db_id], r["pred_sql"])
            except Exception:  # noqa: BLE001 -- any failure means the evaluator zeroed this EM
                failed.add(i)
        unparseable[cond] = failed
        correct = sum(rows[i]["exec"] for i in failed)
        print(f"| {label} | {len(failed)} ({len(failed) / len(rows):.1%}) | {correct} | ≤{correct} |")

    print("\n각 기법이 얻은 Exact Match 중, 베이스라인에서 파싱 실패였을 뿐인 몫:\n")
    print("| 추가한 것 | EM 얻음 | 그중 베이스라인이 파싱 실패 |")
    print("|---|---|---|")
    for base, treat, label in available_pairs(per_example):
        gained = [
            i
            for i, (b, t) in enumerate(zip(per_example[base], per_example[treat]))
            if t["exact"] and not b["exact"]
        ]
        from_unparseable = sum(1 for i in gained if i in unparseable[base])
        share = f"{from_unparseable / len(gained):.0%}" if gained else "-"
        print(f"| {label} | {len(gained)} | {from_unparseable} ({share}) |")


def print_gap_taxonomy(per_example: dict[str, list[dict]]) -> None:
    cloud, local = per_example["cloud_baseline"], per_example["local_ft_rag"]
    gap = [i for i, (c, l) in enumerate(zip(cloud, local)) if c["exec"] and not l["exec"]]
    reverse = sum(1 for c, l in zip(cloud, local) if l["exec"] and not c["exec"])

    print(f"\n## 조건 5가 놓치고 클라우드가 맞힌 {len(gap)}개의 오류 유형\n")
    print(f"반대 방향(조건 5만 맞힘)은 {reverse}개, 순 격차 {len(gap) - reverse}개.\n")

    records = load_predictions("local_ft_rag")
    schema_cache: dict = {}
    buckets: dict[str, list[int]] = {}
    for i in gap:
        r = records[i]
        buckets.setdefault(classify(r["pred_sql"], r["gold_sql"], r["db_id"], schema_cache), []).append(i)

    print("| 오류 유형 | 개수 | 비율 |")
    print("|---|---|---|")
    for name, idxs in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"| {name} | {len(idxs)} | {len(idxs) / len(gap):.1%} |")

    by_hardness = {d: sum(1 for i in gap if local[i]["hardness"] == d) for d in DIFFICULTIES}
    print("\n난이도 분포: " + ", ".join(f"{d} {by_hardness[d]}" for d in DIFFICULTIES))

    print("\n### 유형별 예시\n")
    for name, idxs in sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:5]:
        r = records[idxs[0]]
        print(f"**{name}** — {r['question']}")
        print(f"- gold: `{r['gold_sql']}`")
        print(f"- 조건 5: `{' '.join(r['pred_sql'].split())}`\n")


def main() -> None:
    for eval_set, conditions in EVAL_SETS.items():
        # Conditions are measured in waves, so a listed one may not be scored yet.
        per_example = {
            cond: load_per_example(cond)
            for cond in conditions
            if (RESULTS_DIR / cond / "per_example.jsonl").exists()
        }
        pending = [cond for cond in conditions if cond not in per_example]
        print(f"\n\n# {eval_set}")
        if pending:
            print(f"\n아직 채점되지 않아 제외: {', '.join(pending)}")
        if not per_example:
            continue
        sizes = {len(v) for v in per_example.values()}
        if len(sizes) != 1:
            raise RuntimeError(f"{eval_set}: conditions have different example counts: {sizes}")

        print_difficulty_table(per_example)
        print_difficulty_deltas(per_example)
        print_mcnemar(per_example)
        print_parse_failures(per_example)
        if "cloud_baseline" in per_example and "local_ft_rag" in per_example:
            print_gap_taxonomy(per_example)


if __name__ == "__main__":
    main()
