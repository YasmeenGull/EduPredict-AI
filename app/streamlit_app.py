from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predict import predict  # noqa: E402

st.set_page_config(page_title="EduPredict AI", page_icon="🎓", layout="wide")

st.title("🎓 EduPredict AI")
st.caption("Hybrid ML + Neural Network Student Performance Prediction & Recommendation System")

if not (ROOT / "models" / "random_forest.joblib").exists():
    st.warning("Models are not trained yet. Run `python train.py` first.")
    st.stop()

with st.sidebar:
    st.header("Student Profile")
    age = st.slider("Age", 15, 22, 17)
    medu = st.slider("Mother education (0–4)", 0, 4, 2)
    fedu = st.slider("Father education (0–4)", 0, 4, 2)
    studytime = st.slider("Weekly study time (1–4)", 1, 4, 2)
    failures = st.slider("Past failures", 0, 3, 0)
    absences = st.slider("Absences", 0, 30, 5)
    g1 = st.slider("First-period grade (G1)", 0, 20, 11)
    g2 = st.slider("Second-period grade (G2)", 0, 20, 12)
    famrel = st.slider("Family relationship (1–5)", 1, 5, 4)
    freetime = st.slider("Free time (1–5)", 1, 5, 3)
    goout = st.slider("Going out (1–5)", 1, 5, 3)
    dalc = st.slider("Workday alcohol (1–5)", 1, 5, 1)
    walc = st.slider("Weekend alcohol (1–5)", 1, 5, 1)
    health = st.slider("Health (1–5)", 1, 5, 4)

profile = {
    "school": "GP", "sex": "F", "age": age, "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": medu, "Fedu": fedu, "Mjob": "other", "Fjob": "other",
    "reason": "course", "guardian": "mother", "traveltime": 2,
    "studytime": studytime, "failures": failures, "schoolsup": "no", "famsup": "yes",
    "paid": "no", "activities": "yes", "nursery": "yes", "higher": "yes",
    "internet": "yes", "romantic": "no", "famrel": famrel, "freetime": freetime,
    "goout": goout, "Dalc": dalc, "Walc": walc, "health": health,
    "absences": absences, "G1": g1, "G2": g2,
}

if st.button("🔮 Predict My Performance", type="primary", use_container_width=True):
    score, band, recommendations = predict(profile)
    col1, col2 = st.columns(2)
    col1.metric("Predicted Final Grade", f"{score:.2f} / 20")
    col2.metric("Performance Band", band)

    st.subheader("🤖 Personalized Recommendations")
    for recommendation in recommendations:
        st.write(f"• {recommendation}")

    st.info("This project is an educational ML demonstration, not a definitive assessment of a real student.")

st.divider()

metrics_path = ROOT / "artifacts" / "metrics.json"
if metrics_path.exists():
    metadata = json.loads(metrics_path.read_text())
    st.subheader("📊 Model Evaluation")
    st.json(metadata["metrics"])
