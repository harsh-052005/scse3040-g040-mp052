import json, logging, os, sys, time
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {"ts": self.formatTime(record), "level": record.levelname,
                   "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)  # stdout only, never a file
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("mp052")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

app = FastAPI(title="MP052 Mobile Data Use API")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: int = Field(..., ge=0, le=120)
    screen_hours: float = Field(..., ge=0, le=24)
    apps_installed: int = Field(..., ge=0)
    plan_gb_limit: float = Field(..., gt=0, le=8.5)   # client rule 1
    video_minutes: float = Field(..., ge=0)           # client rule 2
    wifi_hours: float = Field(..., ge=0, le=24)
    phone_tier: Literal["basic", "mid", "flagship"]
    circle: Literal["Urban", "Semi-urban", "Rural"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.perf_counter()
    X = pd.DataFrame([req.model_dump()])
    pred = float(get_model().predict(X)[0])
    logger.info("prediction", extra={"extra_fields": {
        "prediction": round(pred, 4),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2)}})
    return {"prediction": pred}