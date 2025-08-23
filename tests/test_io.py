import pandas as pd

from src.utils.io import save_parquet, load_parquet, save_csv, load_csv, save_to_db, load_from_db
from src import config


def test_save_and_load_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "sample.parquet"

    save_parquet(df, path, index=False)
    df2 = load_parquet(path)

    assert df2 is not None
    assert len(df2) == 3
    assert list(df2.columns) == ["a", "b"]


def test_save_and_load_csv(tmp_path):
    df = pd.DataFrame({"a": [10, 20], "b": [0.1, 0.2]})
    path = tmp_path / "sample.csv"

    save_csv(df, path, index=False)
    df2 = load_csv(path)

    assert df2 is not None
    assert len(df2) == 2
    assert list(df2.columns) == ["a", "b"]


def test_save_and_load_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", str(db_path))

    df = pd.DataFrame({"x": [1, 2, 3], "y": [0.5, 0.6, 0.7]})
    save_to_db(df, table_name="unit_test_table", if_exists="replace")
    df2 = load_from_db("unit_test_table")

    assert df2 is not None
    assert len(df2) == 3
    assert set(df2.columns) == {"x", "y"}
