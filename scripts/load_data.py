"""Smoke-test the data loaders and report dataset sizes.

Usage: uv run python scripts/load_data.py
"""

from rag_text2sql.data import load_spider, load_wikisql, unique_db_ids


def main() -> None:
    wikisql = load_wikisql()
    print("WikiSQL:")
    for split, ds in wikisql.items():
        print(f"  {split}: {len(ds)} examples")

    spider = load_spider()
    print("Spider:")
    for split, ds in spider.items():
        print(f"  {split}: {len(ds)} examples, {len(unique_db_ids(ds))} databases")


if __name__ == "__main__":
    main()
