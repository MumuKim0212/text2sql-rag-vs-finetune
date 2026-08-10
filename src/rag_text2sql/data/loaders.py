"""Loaders for the WikiSQL (pipeline validation) and Spider (main experiment) datasets."""

from datasets import Dataset, DatasetDict, load_dataset

WIKISQL_HF_ID = "Salesforce/wikisql"
SPIDER_HF_ID = "xlangai/spider"


def load_wikisql() -> DatasetDict:
    """Load WikiSQL train/validation/test splits.

    Uses the official Salesforce mirror's pre-converted Parquet branch
    (refs/convert/parquet). The repo's main branch only ships a loading
    script, which recent `datasets` versions refuse to execute; the Parquet
    branch is HF-generated ahead of time from that same script, so it still
    keeps each example's full table (header + rows) needed to execute SQL
    against it later, without running any code locally.
    """
    return load_dataset(WIKISQL_HF_ID, revision="refs/convert/parquet")


def load_spider() -> DatasetDict:
    """Load Spider train/dev splits.

    dev is the fixed held-out evaluation set (Spider's public "validation"
    split, renamed here to match the project's terminology) and must never
    be used for RAG example retrieval or fine-tuning data.
    """
    raw = load_dataset(SPIDER_HF_ID)
    train, dev = raw["train"], raw["validation"]

    overlap = unique_db_ids(train) & unique_db_ids(dev)
    if overlap:
        raise ValueError(
            f"Spider train/dev share {len(overlap)} db_id(s): {sorted(overlap)} "
            "-- this would leak schemas into the held-out evaluation set."
        )

    return DatasetDict({"train": train, "dev": dev})


def unique_db_ids(dataset: Dataset) -> set[str]:
    """Return the distinct Spider db_id values present in a split."""
    return set(dataset["db_id"])
