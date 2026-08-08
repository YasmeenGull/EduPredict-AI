# 🎓 EduPredict AI

**Student Performance Prediction & Personalized Study Recommendation System**

A complete hybrid Artificial Intelligence and Machine Learning project that predicts a student's final academic grade and converts the prediction into personalized study recommendations.

## Why this project?

It combines the requirements of an AI final project and an ML final project in one portfolio-ready system:

- Data collection and preprocessing
- Feature engineering through model pipelines
- Classical ML with Random Forest Regression
- Neural-network concept with an MLP Regressor
- Hybrid ensemble prediction
- Model evaluation using MAE, RMSE and R²
- Result visualization
- Rule-based intelligent recommendation engine
- CLI user interaction
- Streamlit graphical user interaction
- Modular programming
- Automated tests

## Dataset

The default dataset is the **UCI Student Performance** dataset. It contains student academic, demographic, social and school-related attributes and supports both regression and classification-style analysis. The project uses the Mathematics (`student-mat.csv`) data and predicts `G3`, the final grade from 0–20.

The training script automatically downloads the dataset when internet access is available. If the dataset cannot be reached, it creates a deterministic demo dataset so the project remains runnable offline.

## Project architecture

```text
EduPredict-AI/
├── app/
│   └── streamlit_app.py
├── data/
├── models/
├── artifacts/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── recommender.py
│   └── visualization.py
├── tests/
│   └── test_project.py
├── train.py
├── predict.py
├── sample_profile.json
├── requirements.txt
└── README.md
```

## Quick start

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train.py
```

Run a CLI prediction:

```bash
python predict.py --json sample_profile.json
```

Run the interactive app:

```bash
streamlit run app/streamlit_app.py
```

Run tests:

```bash
pytest -q
```

## AI/ML workflow

```text
Raw Student Data
      ↓
Data Cleaning
      ↓
Missing-value Handling + Scaling + One-Hot Encoding
      ↓
Train/Test Split
      ↓
 ┌──────────────────────┐
 │ Random Forest         │
 │ Neural Network (MLP)  │
 └──────────┬───────────┘
            ↓
     Hybrid Prediction
            ↓
  Performance Classification
            ↓
 Personalized Recommendations
            ↓
 CLI / Streamlit UI
```

## Important academic note

The UCI documentation notes that `G1` and `G2` are strongly correlated with `G3`. This project intentionally uses them because they represent earlier-period grades and make the final-grade prediction more useful when those grades are already available. If your instructor wants a stricter early-warning experiment, remove `G1` and `G2` from `NUMERIC_FEATURES` and retrain.

## License / attribution

The dataset is from the UCI Machine Learning Repository. See the dataset page for attribution and licensing information.
