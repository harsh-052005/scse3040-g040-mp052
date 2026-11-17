"""
Data Quality & Anomaly Detection Module for MP052 (Group G040).
Checks for the 7 potential data issues outlined in the Student Handbook:
1. Leaking column
2. Unit mixup
3. Sentinel value
4. Duplicate rows
5. Stuck sensor
6. Target typos
7. Spelling variants
"""

import sys
import yaml
import hashlib
import pandas as pd
import numpy as np


def load_data(params_path="params.yaml"):
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    train_path = params["data"]["raw_train"]
    later_path = params["data"]["raw_later"]
    holdout_path = params["data"]["raw_holdout"]

    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)
    holdout_df = pd.read_csv(holdout_path)
    return train_df, later_df, holdout_df, params


def check_leaking_column(train_df, holdout_df, target_col="gb_per_day"):
    """
    1. Leaking Column: Present in train, missing from holdout, high target correlation.
    """
    candidates = [
        c for c in train_df.columns
        if c not in holdout_df.columns and c != target_col
    ]
    for c in candidates:
        if np.issubdtype(train_df[c].dtype, np.number):
            corr = float(train_df[[c, target_col]].dropna().corr().iloc[0, 1])
            if abs(corr) > 0.95:
                return {
                    "present": True,
                    "column": c,
                    "correlation": round(corr, 4),
                    "reason": f"Missing from holdout_features and {corr:.4f} correlation with {target_col}"
                }
    return {"present": False, "column": None}


def check_unit_mixup(train_df, group_col="circle"):
    """
    2. Unit Mixup: One group recorded one column in another unit (10x, 12x, 60x, 100x, 1000x difference).
    """
    num_cols = ["age", "screen_hours", "apps_installed", "plan_gb_limit", "video_minutes", "wifi_hours"]
    known_factors = [10, 12, 60, 100, 1000]

    for col in num_cols:
        means = train_df.groupby(group_col)[col].mean().to_dict()
        groups = list(means.keys())
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                m1, m2 = means[g1], means[g2]
                if m1 == 0 or m2 == 0:
                    continue
                ratio = m1 / m2 if m1 > m2 else m2 / m1
                for factor in known_factors:
                    if abs(ratio - factor) / factor < 0.15:
                        affected = g1 if m1 > m2 else g2
                        return {
                            "present": True,
                            "column": col,
                            "group_value": affected,
                            "factor": factor
                        }
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(train_df):
    """
    3. Sentinel Value: Special placeholder number typed instead of blank (-1, -999, 999, 9999).
    """
    num_cols = ["age", "screen_hours", "apps_installed", "plan_gb_limit", "video_minutes", "wifi_hours"]
    sentinels = [-1, -99, -999, 999, 9999]

    for col in num_cols:
        vc = train_df[col].value_counts()
        for s in sentinels:
            if s in vc.index and vc[s] > 2:
                return {
                    "present": True,
                    "column": col,
                    "value": int(s),
                    "count": int(vc[s])
                }
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(train_df):
    """
    4. Duplicate Rows: The same record appears multiple times.
    """
    dups = int(train_df.duplicated().sum())
    return {
        "present": dups > 0,
        "count": dups
    }


def check_stuck_sensor(train_df, date_col="date"):
    """
    5. Stuck Sensor: One column shows identical reading for consecutive weeks.
    """
    df = train_df.sort_values(date_col).copy()
    num_cols = ["age", "screen_hours", "apps_installed", "plan_gb_limit", "video_minutes", "wifi_hours"]

    for col in num_cols:
        runs = (df[col] != df[col].shift()).cumsum()
        counts = df.groupby(runs).agg(count=(col, "count"), start=(date_col, "first"), end=(date_col, "last"))
        long_runs = counts[counts["count"] >= 15]
        if not long_runs.empty:
            row = long_runs.iloc[0]
            return {
                "present": True,
                "column": col,
                "start_date": str(row["start"]),
                "end_date": str(row["end"])
            }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(train_df, target_col="gb_per_day"):
    """
    6. Target Typos: Targets typed with an extra zero (10x too big with large gap).
    """
    sorted_target = train_df[target_col].dropna().sort_values().values
    q75 = np.percentile(sorted_target, 75)
    outliers = sorted_target[sorted_target > q75 * 5]

    if len(outliers) > 0 and (outliers.min() / q75 > 5):
        return {
            "present": True,
            "count": int(len(outliers))
        }
    return {"present": False, "count": None}


def check_spelling_variants(train_df, later_df):
    """
    7. Spelling Variants: Categorical column with messy casing/abbreviations. Clean in later.csv.
    """
    text_cols = ["phone_tier", "circle"]
    for col in text_cols:
        train_unique = set(train_df[col].dropna().unique())
        later_unique = set(later_df[col].dropna().unique())
        if len(train_unique) > len(later_unique):
            return {
                "present": True,
                "column": col,
                "true_levels": len(later_unique),
                "levels": sorted(list(later_unique))
            }
    return {"present": False, "column": None, "true_levels": None}


def run_all_checks(params_path="params.yaml"):
    print("=" * 60)
    print("SCSE3040 MLOps Mini Project --- Group G040")
    print("Data Checks & Anomaly Detection (train.csv)")
    print("=" * 60)

    train_df, later_df, holdout_df, params = load_data(params_path)

    # Compute data sha256
    with open(params["data"]["raw_train"], "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"train.csv SHA-256 : {sha256_hash}")
    print(f"Total Rows        : {len(train_df)}")
    print("-" * 60)

    r1 = check_leaking_column(train_df, holdout_df)
    r2 = check_unit_mixup(train_df)
    r3 = check_sentinel_value(train_df)
    r4 = check_duplicate_rows(train_df)
    r5 = check_stuck_sensor(train_df)
    r6 = check_target_typos(train_df)
    r7 = check_spelling_variants(train_df, later_df)

    results = {
        "leaking_column": r1,
        "unit_mixup": r2,
        "sentinel_value": r3,
        "duplicate_rows": r4,
        "stuck_sensor": r5,
        "target_typos": r6,
        "spelling_variants": r7,
    }

    for issue, res in results.items():
        status = "[PRESENT]" if res["present"] else "[ABSENT] "
        print(f"{status} {issue}: {res}")

    print("=" * 60)
    return results


if __name__ == "__main__":
    run_all_checks()
