import os
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
with open(BASE / "academic_advisor.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Academic Advisor")
Level = Literal["Low", "Medium", "High"]


class Student(BaseModel):
    Hours_Studied: float = Field(ge=0, le=80)
    Attendance: float = Field(ge=0, le=100)
    Sleep_Hours: float = Field(ge=0, le=16)
    Previous_Scores: float = Field(ge=0, le=100)
    Tutoring_Sessions: int = Field(ge=0, le=10)
    Motivation_Level: Level
    Parental_Involvement: Level
    Access_to_Resources: Level


def to_grade(score: float) -> str:
    for cutoff, g in [(70, "A"), (60, "B"), (50, "C"), (45, "D"), (40, "E")]:
        if score >= cutoff:
            return g
    return "F"


def advise(s: Student, score: float) -> list[str]:
    tips = []
    if s.Attendance < 75:
        tips.append("Attendance is below 75%. You may be barred from the exam, so fix this first.")
    if s.Hours_Studied < 10:
        tips.append("Raise self-study to at least 10-15 hours a week.")
    if s.Sleep_Hours < 6:
        tips.append("Sleep under 6 hours hurts recall. Aim for 7-8.")
    if s.Previous_Scores < 55:
        tips.append("Weak previous results: revisit the foundations and see your level adviser.")
    if s.Tutoring_Sessions == 0 and score < 70:
        tips.append("Try peer tutoring or a study group.")
    if s.Motivation_Level == "Low":
        tips.append("Set small weekly goals and track them to rebuild momentum.")
    if score < 50:
        tips.append("Predicted score is below a C. Prioritise this course in your revision timetable.")
    if not tips:
        tips.append("You're on track. Keep your current routine.")
    return tips


def ai_plan(s: Student, score: float, grade: str, tips: list[str]) -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None  # app still works without the LLM
    try:
        from groq import Groq

        client = Groq(api_key=key)
        prompt = (
            f"A student's profile: {s.model_dump()}.\n"
            f"A trained model predicts an exam score of {score:.1f} (grade {grade}).\n"
            f"Flagged issues: {tips}.\n"
            "Write a short, encouraging 5-line weekly study plan. "
            "Use only the numbers given. Do not invent facts."
        )
        r = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1000,
            reasoning_effort="low",
        )
        return r.choices[0].message.content or None
    except Exception:
        return None  # fall back to rule-based advice


@app.post("/predict")
def predict(s: Student):
    row = pd.DataFrame([s.model_dump()])
    score = min(float(model.predict(row)[0]), 100.0)
    grade = to_grade(score)
    tips = advise(s, score)
    return {"predicted_score": round(score, 1), "grade": grade, "advice": tips,
            "ai_plan": ai_plan(s, score, grade, tips)}


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Academic Advisor</title>
<style>body{font-family:sans-serif;max-width:480px;margin:2rem auto;padding:0 1rem}
label{display:block;margin-top:.7rem}input,select{width:100%;padding:.4rem}button{margin-top:1rem;padding:.6rem 1rem}
#out{margin-top:1.5rem;padding:1rem;background:#eef;border-radius:8px;display:none}</style></head><body>
<h2>AI Academic Advisor</h2><form id=f></form><button onclick=go()>Get advice</button><div id=out></div>
<script>
const N={Hours_Studied:"Study hours / week",Attendance:"Attendance %",Sleep_Hours:"Sleep hours / night",
Previous_Scores:"Previous score %",Tutoring_Sessions:"Tutoring sessions / month"};
const S={Motivation_Level:"Motivation",Parental_Involvement:"Parental involvement",Access_to_Resources:"Access to resources"};
f.innerHTML=Object.entries(N).map(([k,v])=>`<label>${v}<input id=${k} type=number step=any></label>`).join("")+
Object.entries(S).map(([k,v])=>`<label>${v}<select id=${k}><option>Low<option selected>Medium<option>High</select></label>`).join("");
async function go(){const b={};for(const k in N)b[k]=parseFloat(document.getElementById(k).value);
for(const k in S)b[k]=document.getElementById(k).value;
const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)});
const d=await r.json();out.style.display="block";
out.innerHTML=r.ok?`<b>Predicted score: ${d.predicted_score} (Grade ${d.grade})</b><ul>${d.advice.map(a=>`<li>${a}</li>`).join("")}</ul>${d.ai_plan?"<p><b>AI study plan:</b><br>"+d.ai_plan.replace(/\\n/g,"<br>")+"</p>":""}`:"Check your inputs.";}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
