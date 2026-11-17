import pandas as pd
from src.mp052.clean import clean_dataset, clean_phone_tier_series


def test_clean_phone_tier_variants():
    raw_series = pd.Series(["BASIC", " bas. ", "mid.", "FLAGSHIP", "fla.", " mid ", "basic"])
    cleaned = clean_phone_tier_series(raw_series)
    expected = ["basic", "basic", "mid", "flagship", "flagship", "mid", "basic"]
    assert list(cleaned) == expected


def test_clean_dataset_removes_duplicates():
    df = pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "age": [25, 25, 30],
        "phone_tier": ["basic", "basic", "mid"],
        "gb_per_day": [2.5, 2.5, 3.0]
    })
    cleaned = clean_dataset(df, is_training=True)
    assert len(cleaned) == 2
    assert list(cleaned["record_id"]) == ["R1", "R2"]


def test_clean_dataset_drops_leaking_column():
    df = pd.DataFrame({
        "record_id": ["R1", "R2"],
        "age": [25, 30],
        "gb_from_monthly_bill": [2.4, 2.9],
        "gb_per_day": [2.5, 3.0]
    })
    cleaned = clean_dataset(df)
    assert "gb_from_monthly_bill" not in cleaned.columns
    assert "gb_per_day" in cleaned.columns
