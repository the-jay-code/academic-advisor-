# 🎓 AI Academic Advisor

An academic advisor system that predicts a student's exam score from their study habits, explains what to improve, and lets students chat with it in plain language. Built for **CSC 303** by Polycarp Joshua (Computer Science, Veritas University Abuja).

It combines **standard software engineering practice** (Git/GitHub, virtual environments, input validation, automated tests) with **AI techniques** (a trained machine-learning model plus an LLM that explains results).

## How it works

```
Student ──► Chat (Streamlit) ──► LLM (Groq) ──► calls trained model (.pkl)
   │                                  ▲                 │
   └──► Web form (FastAPI) ───────────┴── rules advice ◄┘
```

1. **ML model:** a scikit-learn pipeline trained in Google Colab on Kaggle's *Student Performance Factors* dataset, saved as `academic_advisor.pkl`.
2. **Rules layer:** simple, transparent checks (e.g. attendance below 75% risks exam ineligibility).
3. **LLM layer:** a Groq-hosted model writes a short study plan and powers the chat bot. It never calculates the score itself; it calls the trained model as a tool and is told to answer only from the project's data facts.

## Features

- 🔮 Exam score prediction from 8 inputs (study hours, attendance, sleep, previous score, tutoring, motivation, parental involvement, access to resources)
- 📋 Rule-based advice for risky habits
- 🤖 AI-written weekly study plan (optional, needs a Groq key)
- 💬 Chat bot: describe yourself in plain language and get a prediction and advice
- ✅ Input validation (Pydantic) and automated tests (pytest)

## Model results

Data: 6,607 students (Kaggle, *Student Performance Factors*). 80/20 train/test split.

| Model | MAE (marks) | R² |
|---|---|---|
| **Linear Regression (selected)** | **0.90** | **0.717** |
| Random Forest | 1.19 | 0.633 |

Strongest relationships with exam score: attendance (0.58), hours studied (0.45), previous scores (0.18), tutoring sessions (0.16). Sleep showed almost no relationship (-0.02) in this data.

The simpler linear model won, which is consistent with this dataset being close to linear.

## Tech stack

Python 3.12 · pandas · scikit-learn 1.6.1 · FastAPI · Streamlit · Groq API (`openai/gpt-oss-20b`) · pytest · Git/GitHub · Google Colab · VS Code

## Project structure

```
├── academic_advisor.ipynb   # Colab notebook: data exploration, training, model comparison
├── academic_advisor.pkl     # Trained model (scikit-learn pipeline)
├── main.py                  # FastAPI app: /predict API, web form, rules, LLM study plan
├── chat_app.py              # Streamlit chat bot
├── conftest.py              # Test setup (prevents real LLM calls in tests)
├── tests/test_api.py        # pytest tests for the API
├── requirements.txt
└── README.md
```

## Run it locally

```powershell
git clone https://github.com/the-jay-code/academic-advisor-.git
cd academic-advisor-
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The LLM features need a free [Groq](https://console.groq.com) API key. Set it in your terminal (never commit it):

```powershell
$env:GROQ_API_KEY="your_key_here"
```

**Web app and API** (works without a key, but with no AI study plan):

```powershell
uvicorn main:app --reload
```
Open http://127.0.0.1:8000 for the form, or http://127.0.0.1:8000/docs to try the API.

**Chat bot** (needs the key):

```powershell
streamlit run chat_app.py
```
Open http://localhost:8501 and try: *"I study 8 hours a week, attend 60% of classes, sleep 5 hours, scored 55% before and have no tutoring."*

### Example API call

`POST /predict`
```json
{
  "Hours_Studied": 20, "Attendance": 92, "Sleep_Hours": 7,
  "Previous_Scores": 80, "Tutoring_Sessions": 2,
  "Motivation_Level": "High", "Parental_Involvement": "Medium",
  "Access_to_Resources": "High"
}
```

## Tests

```powershell
python -m pytest -q
```
Five tests cover a valid prediction, a warning for a weak student, rejection of invalid input, and the optional LLM study plan (with and without an API key).

## Screenshots

<!-- Add your own: create a screenshots/ folder, add images, then uncomment these -->
<!-- ![Web form](screenshots/web-form.png) -->
<!-- ![Chat bot](screenshots/chat.png) -->
<!-- ![Tests passing](screenshots/tests.png) -->

## Limitations

- The dataset is not Nigerian-university data and is partly synthetic, and most exam scores sit between 60 and 80, so predictions are indicative only.
- Advice is general guidance and does not replace a real academic adviser.
- The chat bot answers only from the facts it is given and the prediction tool, but LLMs can still make mistakes.

## Author

Polycarp Joshua · Computer Science, Veritas University Abuja · Matric No: [add yours] · CSC 303 AI project

## Data source

[Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) by lainguyn123, via Kaggle.