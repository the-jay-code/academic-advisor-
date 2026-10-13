import json
import os

import pandas as pd
import streamlit as st
from groq import Groq

from main import Student, advise, model  # reuse the app's logic

MODEL = "openai/gpt-oss-20b"
LEVELS = ["Motivation_Level", "Parental_Involvement", "Access_to_Resources"]

# Facts taken from the Colab analysis of Kaggle's Student Performance Factors dataset
FACTS = """
DATASET: Kaggle 'Student Performance Factors', 6,607 students, exam scores mostly 60-80.
Correlation with Exam_Score: Attendance 0.58 (strongest), Hours_Studied 0.45,
Previous_Scores 0.18, Tutoring_Sessions 0.16, Physical_Activity 0.03, Sleep_Hours -0.02 (almost none).
MODEL: linear regression pipeline, R2 = 0.717, mean absolute error = 0.9 marks (random forest was worse: R2 0.633).
RULES used by the advisor: attendance below 75% risks exam ineligibility; aim for 10-15 study hours/week; 7-8 hours sleep.
Limits: dataset is not Nigerian-university data and is partly synthetic, so predictions are indicative only.
"""

SYSTEM = f"""You are a friendly academic advisor chatbot for university students.
Answer ONLY from the facts below and from the predict_exam_score tool. Never invent statistics.
If a question is not covered by the facts, say you don't have data on it.
When the student describes themselves, call predict_exam_score. It needs: Hours_Studied (per week),
Attendance (%), Sleep_Hours (per night), Previous_Scores (%), Tutoring_Sessions (per month).
If any of those five are missing, ask for them before calling the tool. For Motivation, Parental
involvement and Access to resources, assume Medium if not mentioned and say you assumed it.
Do not give letter grades: in this dataset most scores fall between 60 and 80, so compare a
prediction with that range. After the tool returns, explain the result and the advice in a warm, concise way (max ~8 lines).
{FACTS}"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "predict_exam_score",
        "description": "Predict a student's exam score and get rule-based advice using the trained model.",
        "parameters": {
            "type": "object",
            "properties": {
                "Hours_Studied": {"type": "number", "description": "Study hours per week"},
                "Attendance": {"type": "number", "description": "Attendance percentage 0-100"},
                "Sleep_Hours": {"type": "number", "description": "Hours of sleep per night"},
                "Previous_Scores": {"type": "number", "description": "Previous score percentage 0-100"},
                "Tutoring_Sessions": {"type": "integer", "description": "Tutoring sessions per month"},
                "Motivation_Level": {"type": "string", "enum": ["Low", "Medium", "High"]},
                "Parental_Involvement": {"type": "string", "enum": ["Low", "Medium", "High"]},
                "Access_to_Resources": {"type": "string", "enum": ["Low", "Medium", "High"]},
            },
            "required": ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Tutoring_Sessions"],
        },
    },
}]


def predict_exam_score(args: dict) -> dict:
    args = dict(args)
    assumed = [k for k in LEVELS if k not in args]
    for k in assumed:
        args[k] = "Medium"
    try:
        s = Student(**args)
    except Exception as e:  # invalid or missing values go back to the LLM
        return {"error": str(e)}
    score = min(float(model.predict(pd.DataFrame([s.model_dump()]))[0]), 100.0)
    return {"predicted_score": round(score, 1),
            "advice": advise(s, score), "assumed_medium_for": assumed}


def respond(client, history: list[dict]) -> str:
    """Send the chat to the LLM, run the prediction tool if it asks, return the final answer."""
    messages = [{"role": "system", "content": SYSTEM}] + history
    for _ in range(3):
        r = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
            max_completion_tokens=1500, reasoning_effort="low",
        )
        msg = r.choices[0].message
        if not msg.tool_calls:
            return msg.content or "Sorry, I couldn't come up with an answer. Try rephrasing."
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = (predict_exam_score(args) if tc.function.name == "predict_exam_score"
                      else {"error": "unknown tool"})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Sorry, something went wrong. Please try again."


def main():
    st.set_page_config(page_title="AI Academic Advisor", page_icon="🎓")
    st.title("🎓 AI Academic Advisor")
    st.caption("Chat about your study habits. Predictions come from a model trained on Kaggle data.")

    key = os.getenv("GROQ_API_KEY")
    if not key:
        st.error("GROQ_API_KEY is not set. Set it in the terminal and restart the app.")
        st.stop()
    client = Groq(api_key=key)

    if "history" not in st.session_state:
        st.session_state.history = [{"role": "assistant", "content":
            "Hi! Tell me about your study habits, for example: *I study 8 hours a week, attend "
            "60% of classes, sleep 5 hours, scored 55% before and have no tutoring.* "
            "Or ask what matters most for exam scores."}]

    for m in st.session_state.history:
        st.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Ask the advisor..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = respond(client, st.session_state.history)
                except Exception as e:
                    answer = f"Sorry, the AI service returned an error: {type(e).__name__}: {e}"
            st.write(answer)
        st.session_state.history.append({"role": "assistant", "content": answer})


main()