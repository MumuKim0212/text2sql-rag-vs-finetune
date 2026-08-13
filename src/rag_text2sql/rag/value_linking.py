"""Value linking: surface the database cell values a question mentions.

The base model writes `WHERE PetType = 'Cat'` where the column actually holds
'cat'; SQLite's `=` is case-sensitive, so the query returns no rows and scores
wrong. Only the database itself can settle the exact literal.

Reading cell values from the database being queried is not a leak of the
evaluation set: the rule in project-text2sql-brief.md covers dev questions and
gold SQL, and any real deployment has the target database in hand. This is also
the one thing fine-tuning cannot substitute for, since dev DB contents appear
in no training split.
"""

import re
import sqlite3
from pathlib import Path

TEXT_TYPES = ("TEXT", "CHAR", "VARCHAR", "CLOB")
MAX_DISTINCT_PER_COLUMN = 200  # bounds the scan on wide tables like wta_1
MAX_VALUE_LEN = 60
MAX_HITS = 20


def load_db_values(sqlite_path: str | Path) -> list[tuple[str, str, str]]:
    """Distinct text values in a database, as (table, column, value)."""
    con = sqlite3.connect(str(sqlite_path))
    con.text_factory = lambda b: b.decode(errors="ignore")
    cursor = con.cursor()

    values: list[tuple[str, str, str]] = []
    tables = [
        row[0]
        for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if not row[0].startswith("sqlite_")
    ]
    for table in tables:
        try:
            columns = cursor.execute(f'PRAGMA table_info("{table}")').fetchall()
        except sqlite3.Error:
            continue
        for _, column, column_type, *_ in columns:
            if not any(t in (column_type or "").upper() for t in TEXT_TYPES):
                continue
            try:
                rows = cursor.execute(
                    f'SELECT DISTINCT "{column}" FROM "{table}" LIMIT {MAX_DISTINCT_PER_COLUMN}'
                ).fetchall()
            except sqlite3.Error:
                continue  # generated/virtual columns and malformed tables are skipped
            for (value,) in rows:
                if isinstance(value, str) and 2 <= len(value.strip()) <= MAX_VALUE_LEN:
                    values.append((table, column, value.strip()))

    con.close()
    return values


# Questions inflect the values they mention: "cats" for 'cat', "Asian" for 'Asia',
# "European" for 'Europe'. A closed list rather than a stemmer -- it has to add
# recall without letting 'car' fire on "caring", and these cover what Spider dev
# actually asks. Longest first so the alternation prefers the fuller suffix.
SUFFIXES = ("ians", "ian", "ese", "ish", "ans", "an", "es", "s", "n")


def _forms(value: str) -> set[str]:
    """The spellings of a value worth looking for in a question.

    Some values are written closed-up where the question spaces them out
    ('NorthCarolina' vs "North Carolina"), so a camel-case split is tried too.
    """
    lowered = value.lower()
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value).lower()
    return {lowered, spaced}


def _pattern(form: str) -> str:
    """A regex matching `form` as a whole word, tolerating an inflected ending.

    `\\b` is wrong here: it needs a word character on its side, so a value ending
    in punctuation -- 'amc hornet sportabout (sw)' -- never matched even when the
    question contained it verbatim. Lookarounds for a word character have the
    same effect on alphanumeric values and also work on the rest.
    """
    suffixes = "|".join(SUFFIXES) if form[-1:].isalpha() else ""
    tail = f"(?:{suffixes})?" if suffixes else ""
    return r"(?<!\w)" + re.escape(form) + tail + r"(?!\w)"


def match_values(
    question: str, values: list[tuple[str, str, str]], max_hits: int = MAX_HITS
) -> list[tuple[str, str, str]]:
    """The values occurring in `question` as whole words, ignoring case.

    Whole-word matching keeps 'cat' from firing on "category". Matching
    case-insensitively is the point: a hit that differs from the question's own
    casing is exactly the literal the model would otherwise get wrong.
    """
    lowered = question.lower()
    hits = {
        (table, column, value)
        for table, column, value in values
        if any(re.search(_pattern(form), lowered) for form in _forms(value))
    }
    # Longest first -- a longer value is a more specific match than a substring of it.
    return sorted(hits, key=lambda hit: (-len(hit[2]), hit))[:max_hits]
