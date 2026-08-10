from rag_text2sql.data import load_spider, load_wikisql, unique_db_ids


def test_load_wikisql_has_expected_splits():
    ds = load_wikisql()
    assert set(ds.keys()) == {"train", "validation", "test"}
    assert len(ds["train"]) > 0


def test_load_spider_dev_is_held_out_and_disjoint():
    ds = load_spider()
    assert set(ds.keys()) == {"train", "dev"}

    assert unique_db_ids(ds["train"]).isdisjoint(unique_db_ids(ds["dev"]))
    assert len(ds["dev"]) == 1034
    assert len(unique_db_ids(ds["dev"])) == 20
