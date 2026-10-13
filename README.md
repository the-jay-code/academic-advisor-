# AI Academic Advisor

Predicts a student's exam score from study habits and gives advice. Built for CSC 303.

- **Data:** Kaggle, Student Performance Factors (6,607 students)
- **Model:** scikit-learn linear regression pipeline (R2 0.717, MAE 0.9), trained in Google Colab (academic_advisor.ipynb) and saved as academic_advisor.pkl
- **Web app:** FastAPI (main.py) with a form, a /predict API and rule-based advice, plus an optional Groq LLM study plan
- **Chat bot:** Streamlit (chat_app.py). Talk to the advisor in plain language; the LLM calls the model as a tool
- **Tests:** pytest (tests/test_api.py)

## Run it
pip install -r requirements.txt
set GROQ_API_KEY (PowerShell: $env:GROQ_API_KEY="your_key")
uvicorn main:app --reload      # web form at http://127.0.0.1:8000
streamlit run chat_app.py      # chat bot at http://localhost:8501
pytest -q

## Limitations
The dataset is not Nigerian-university data and is partly synthetic, so predictions are indicative only. LLM advice is general and does not replace a real academic adviser.
