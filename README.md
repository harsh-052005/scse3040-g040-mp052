# SCSE3040 MLOps Mini Project --- Group G040

## Problem: MP052 --- Mobile Data Use (Indore)

### Group Details
- **Group ID:** G040
- **Problem ID:** MP052 (Mobile Data Use - Indore)
- **Batch:** 91
- **Faculty:** Dr. Amit Soni

### Team Members
| Enrolment No. | Name | Role |
| :--- | :--- | :--- |
| **S24CSEU0766** | HARSH. | Data Lead |
| **S24CSEU2276** | YashSinghPratap | Service Lead |
| **S24CSEU0388** | SALONI. | Model Lead |
| **S24CSEU0706** | ASHUTOSHSINGHKUMAR | Ops & Monitoring Lead |

---

## Project Overview
A complete end-to-end MLOps pipeline for Indore Telecom Circle Office to predict daily mobile data consumption (`gb_per_day`) based on user profiles and usage statistics.

### Pipeline Components
1. **Data Versioning (DVC):** Raw datasets tracked with DVC (`train.csv.dvc`, `later.csv.dvc`).
2. **Data Quality & Cleaning:** Automated detection and fixing of 7 potential data anomalies.
3. **Experiment Tracking & Model Registry:** Tracked experiments via MLflow (`sqlite:///mlflow.db`) with `@champion` alias.
4. **REST API Service:** FastAPI service exposing `/predict` and `/health` with strict Pydantic validation rules.
5. **Containerization & CI/CD:** Docker container (`mp052-api:1.0`) and automated GitHub Actions test pipeline.
6. **Kubernetes Deployment:** K8s deployment (`mp052-api`) and service (`mp052-svc`) manifests with health probes.
7. **Production Drift Monitoring:** Population Stability Index (PSI) calculation and automated retraining (v2) comparison.

---

## Repository Structure
```text
├── README.md                          # Group, problem and team member details
├── requirements.txt                   # Pinned project dependencies
├── params.yaml                        # Central configuration parameters
├── Dockerfile                         # Container definition
├── .dockerignore                      # Docker ignore rules
├── .gitignore                         # Git ignore rules
├── mlflow.db                          # MLflow tracking database
├── src/
│   └── mp052/                         # Modular package code
│       ├── __init__.py
│       ├── data_checks.py             # Data quality & issue detection
│       ├── clean.py                   # Data preprocessing pipeline
│       ├── train.py                   # MLflow training & registry
│       └── monitor.py                 # PSI drift & retraining comparison
├── tests/
│   ├── __init__.py
│   ├── test_data.py                   # Unit tests for data cleaning
│   └── test_api.py                    # API tests with FastAPI TestClient
├── app/
│   └── main.py                        # FastAPI prediction service
├── models/
│   └── model.joblib                   # Champion model artifact
├── data/
│   └── raw/
│       ├── train.csv.dvc              # DVC metadata for train.csv
│       ├── later.csv.dvc              # DVC metadata for later.csv
│       └── holdout_features.csv       # Holdout evaluation set
├── reports/
│   └── findings.yaml                  # Evaluation report form
├── predictions/
│   └── holdout_predictions.csv        # Champion model predictions on holdout
└── k8s/
    ├── deployment.yaml                # K8s Deployment manifest
    └── service.yaml                   # K8s Service manifest
```

---

## Setup & Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Data Issues Detection
```bash
python3 -m src.mp052.data_checks
```

### 3. Run Tests
```bash
pytest
```
