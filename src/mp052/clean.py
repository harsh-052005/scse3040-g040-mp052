"""
Data Cleaning and Preprocessing Pipeline for MP052 (Group G040).
Applies fixes for identified data issues:
- Drops leaking column (gb_from_monthly_bill)
- Removes duplicate rows
- Standardizes spelling variants in phone_tier
"""

import os
import yaml
import pandas as pd


def clean_phone_tier_series(series: pd.Series) -> pd.Series:
    """Standardizes phone_tier text variants into ['basic', 'mid', 'flagship']."""
    def _clean_val(v):
        if not isinstance(v, str):
            return v
        val = v.strip().lower()
        if "bas" in val:
            return "basic"
        elif "fla" in val:
            return "flagship"
        elif "mid" in val:
            return "mid"
        return val

    return series.apply(_clean_val)


def clean_dataset(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Cleans raw dataframe by removing duplicates, leaking columns,
    and standardizing categorical spelling.
    """
    df = df.copy()

    # 1. Remove duplicate rows if in training
    if is_training:
        df = df.drop_duplicates().reset_index(drop=True)

    # 2. Drop leaking column if present
    if "gb_from_monthly_bill" in df.columns:
        df = df.drop(columns=["gb_from_monthly_bill"])

    # 3. Clean categorical spelling variants
    if "phone_tier" in df.columns:
        df["phone_tier"] = clean_phone_tier_series(df["phone_tier"])

    return df


def run_pipeline(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    raw_train_path = params["data"]["raw_train"]
    raw_later_path = params["data"]["raw_later"]
    processed_train_path = params["data"]["processed_train"]
    processed_later_path = params["data"]["processed_later"]

    os.makedirs(os.path.dirname(processed_train_path), exist_ok=True)

    # Clean train
    train_df = pd.read_csv(raw_train_path)
    clean_train_df = clean_dataset(train_df, is_training=True)
    clean_train_df.to_csv(processed_train_path, index=False)
    print(f"Cleaned train: {len(train_df)} -> {len(clean_train_df)} rows saved to {processed_train_path}")

    # Clean later
    later_df = pd.read_csv(raw_later_path)
    clean_later_df = clean_dataset(later_df, is_training=False)
    clean_later_df.to_csv(processed_later_path, index=False)
    print(f"Cleaned later: {len(later_df)} -> {len(clean_later_df)} rows saved to {processed_later_path}")


if __name__ == "__main__":
    run_pipeline()
