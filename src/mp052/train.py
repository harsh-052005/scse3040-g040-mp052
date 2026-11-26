"""Train v1 models, track them with MLflow, register the best as @champion.

Run from the repo root:  python -m src.mp052.train
"""
import hashlib
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.mp052.clean import clean_dataset

MODEL_TYPES = {
    "Ridge": Ridge,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
}


def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(model_type, model_params, numerical, categorical, random_state=42):
    """Full pipeline: scale numbers, one-hot the text columns, then the regressor.
    The API can pass a raw one-row DataFrame straight to .predict()."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    cls = MODEL_TYPES[model_type]
    kwargs = dict(model_params)
    if model_type != "Ridge":
        kwargs["random_state"] = random_state
    return Pipeline([("pre", pre), ("model", cls(**kwargs))])


def main(params_path="params.yaml"):
    params = load_params(params_path)
    feats = params["features"]
    tcfg = params["train"]
    numerical, categorical = feats["numerical"], feats["categorical"]
    target = feats["target"]
    columns = numerical + categorical

    raw_path = params["data"]["raw_train"]
    data_sha = sha256_of_file(raw_path)  # hash of the ORIGINAL file
    df = clean_dataset(pd.read_csv(raw_path), is_training=True)
    X, y = df[columns], df[target]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=tcfg["test_size"], random_state=tcfg["random_state"]
    )

    mlflow.set_tracking_uri(tcfg["tracking_uri"])
    mlflow.set_experiment(tcfg["experiment_name"])

    results = []  # (mae, run_id, fitted_pipeline)
    for cfg in params["models"]:
        pipe = build_pipeline(
            cfg["type"], cfg["params"], numerical, categorical, tcfg["random_state"]
        )
        with mlflow.start_run(run_name=cfg["name"]) as run:
            pipe.fit(X_tr, y_tr)
            mae = mean_absolute_error(y_va, pipe.predict(X_va))
            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("model_type", cfg["type"])
            mlflow.log_params({f"hp_{k}": v for k, v in cfg["params"].items()})
            mlflow.log_param("n_train_rows", len(X_tr))
            mlflow.log_metric("mae_valid", float(mae))
            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                signature=infer_signature(X_tr, pipe.predict(X_tr)),
                serialization_format="cloudpickle",
            )
            results.append((mae, run.info.run_id, pipe))
            print(f"{cfg['name']:<22} mae_valid={mae:.4f}  run_id={run.info.run_id}")

    best_mae, best_run_id, best_pipe = min(results, key=lambda r: r[0])

    name = tcfg["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{best_run_id}/model", name)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)

    model_file = params["paths"]["model_file"]
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    joblib.dump(best_pipe, model_file)

    print("\n=== copy these into reports/findings.yaml (tracking:) ===")
    print(f"experiment_name: {tcfg['experiment_name']}")
    print(f"best_run_id: {best_run_id}")
    print(f"registered_model_name: {name}")
    print(f"champion_version: {mv.version}")
    print(f"best mae_valid: {best_mae:.4f}")


if __name__ == "__main__":
    main()