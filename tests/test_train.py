import numpy as np
import pandas as pd

from src.mp052.train import build_pipeline, sha256_of_file

NUM = ["age", "screen_hours", "apps_installed", "plan_gb_limit", "video_minutes", "wifi_hours"]
CAT = ["phone_tier", "circle"]


def _fake_df(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(16, 70, n),
            "screen_hours": rng.uniform(0.5, 12, n),
            "apps_installed": rng.integers(10, 200, n),
            "plan_gb_limit": rng.uniform(1, 5, n),
            "video_minutes": rng.integers(1, 300, n),
            "wifi_hours": rng.uniform(0, 16, n),
            "phone_tier": rng.choice(["basic", "mid", "flagship"], n),
            "circle": rng.choice(["Urban", "Semi-urban", "Rural"], n),
        }
    )


def test_pipeline_fits_and_predicts_one_value_per_row():
    X = _fake_df()
    y = X["screen_hours"] * 0.4 + 1.0
    pipe = build_pipeline("Ridge", {"alpha": 1.0}, NUM, CAT)
    pipe.fit(X, y)
    assert pipe.predict(X.head(5)).shape == (5,)


def test_pipeline_ignores_unseen_category():
    X = _fake_df()
    y = X["screen_hours"] * 0.4 + 1.0
    pipe = build_pipeline("RandomForestRegressor", {"n_estimators": 5, "max_depth": 3}, NUM, CAT)
    pipe.fit(X, y)
    row = X.head(1).copy()
    row["circle"] = "Metro"
    assert np.isfinite(pipe.predict(row)[0])


def test_sha256_of_file_is_stable(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes(b"abc")
    assert sha256_of_file(str(p)) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )