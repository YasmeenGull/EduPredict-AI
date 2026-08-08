# 🎓 EduPredict AI

### Intelligent Student Performance Prediction & Personalized Learning System

**EduPredict AI** is a hybrid AI/ML application that predicts student academic performance and provides personalized study recommendations based on the prediction.

The system combines **Random Forest Regression, MLP Neural Network, and Ensemble Learning** with a rule-based recommendation engine to transform student data into actionable academic insights.

### ✨ Key Features

* 📊 Data preprocessing & feature engineering
* 🤖 Random Forest & MLP Regression
* 🔗 Hybrid ensemble prediction
* 📈 MAE, RMSE & R² evaluation
* 💡 Personalized study recommendations
* 🌐 Streamlit web application
* 💻 CLI interface
* 🧪 Automated testing
* 🧩 Modular architecture

**Tech Stack:** Python • Pandas • NumPy • Scikit-learn • Matplotlib • Streamlit • Pytest


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
