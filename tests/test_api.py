from fastapi.testclient import TestClient
import app.main as m

class DummyModel:
    def predict(self, X):
        return [3.14]

m.get_model = lambda: DummyModel()
client = TestClient(m.app)

VALID = dict(age=30, screen_hours=5.0, apps_installed=80, plan_gb_limit=3.0,
             video_minutes=120, wifi_hours=6.0, phone_tier="mid", circle="Urban")

def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

def test_predict_ok():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200 and r.json()["prediction"] == 3.14

def test_plan_limit_above_8_5_rejected():
    assert client.post("/predict", json={**VALID, "plan_gb_limit": 9.0}).status_code == 422

def test_negative_video_minutes_rejected():
    assert client.post("/predict", json={**VALID, "video_minutes": -5}).status_code == 422