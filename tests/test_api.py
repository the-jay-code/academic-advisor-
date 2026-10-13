from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
GOOD = dict(Hours_Studied=20, Attendance=92, Sleep_Hours=7, Previous_Scores=80,
            Tutoring_Sessions=2, Motivation_Level="High",
            Parental_Involvement="Medium", Access_to_Resources="High")


def test_predict_ok():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert 0 <= r.json()["predicted_score"] <= 100


def test_weak_student_gets_warning():
    bad = dict(GOOD, Attendance=50, Hours_Studied=2)
    advice = client.post("/predict", json=bad).json()["advice"]
    assert any("Attendance" in a for a in advice)


def test_invalid_input_rejected():
    assert client.post("/predict", json=dict(GOOD, Motivation_Level="Huge")).status_code == 422


def test_ai_plan_is_none_without_key():
    assert client.post("/predict", json=GOOD).json()["ai_plan"] is None


def test_ai_plan_used_when_llm_available(monkeypatch):
    import main
    monkeypatch.setattr(main, "ai_plan", lambda *a, **k: "Study plan here")
    assert client.post("/predict", json=GOOD).json()["ai_plan"] == "Study plan here"
