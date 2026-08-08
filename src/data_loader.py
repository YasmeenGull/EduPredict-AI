from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .config import DATA_DIR

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student.zip"
LOCAL_FILE = DATA_DIR / "student-mat.csv"


def _generate_fallback_dataset(n: int = 1200) -> pd.DataFrame:
    """Create a deterministic demo dataset if UCI is temporarily unreachable."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "school": rng.choice(["GP", "MS"], n, p=[0.88, 0.12]),
        "sex": rng.choice(["F", "M"], n),
        "age": rng.integers(15, 23, n),
        "address": rng.choice(["U", "R"], n, p=[0.75, 0.25]),
        "famsize": rng.choice(["LE3", "GT3"], n),
        "Pstatus": rng.choice(["T", "A"], n, p=[0.9, 0.1]),
        "Medu": rng.integers(0, 5, n),
        "Fedu": rng.integers(0, 5, n),
        "Mjob": rng.choice(["teacher", "health", "services", "at_home", "other"], n),
        "Fjob": rng.choice(["teacher", "health", "services", "at_home", "other"], n),
        "reason": rng.choice(["home", "reputation", "course", "other"], n),
        "guardian": rng.choice(["mother", "father", "other"], n, p=[0.7, 0.25, 0.05]),
        "traveltime": rng.integers(1, 5, n),
        "studytime": rng.integers(1, 5, n),
        "failures": rng.choice([0, 1, 2, 3], n, p=[0.65, 0.22, 0.10, 0.03]),
        "schoolsup": rng.choice(["yes", "no"], n, p=[0.15, 0.85]),
        "famsup": rng.choice(["yes", "no"], n, p=[0.65, 0.35]),
        "paid": rng.choice(["yes", "no"], n, p=[0.35, 0.65]),
        "activities": rng.choice(["yes", "no"], n),
        "nursery": rng.choice(["yes", "no"], n, p=[0.8, 0.2]),
        "higher": rng.choice(["yes", "no"], n, p=[0.9, 0.1]),
        "internet": rng.choice(["yes", "no"], n, p=[0.85, 0.15]),
        "romantic": rng.choice(["yes", "no"], n),
        "famrel": rng.integers(1, 6, n),
        "freetime": rng.integers(1, 6, n),
        "goout": rng.integers(1, 6, n),
        "Dalc": rng.integers(1, 6, n),
        "Walc": rng.integers(1, 6, n),
        "health": rng.integers(1, 6, n),
        "absences": np.clip(rng.poisson(5, n), 0, 30),
    })
    base = (
        8.0 + 1.05 * df["studytime"] + 0.42 * df["Medu"] + 0.35 * df["Fedu"]
        - 0.85 * df["failures"] - 0.07 * df["absences"]
        - 0.30 * df["traveltime"] + 0.15 * df["famrel"]
        + rng.normal(0, 1.6, n)
    )
    df["G1"] = np.clip(base + rng.normal(0, 1.2, n), 4, 18).round().astype(int)
    df["G2"] = np.clip(base + 0.5 + rng.normal(0, 1.0, n), 4, 19).round().astype(int)
    df["G3"] = np.clip(
        0.25 * base + 0.38 * df["G1"] + 0.40 * df["G2"]
        + rng.normal(0, 1.2, n), 0, 20
    ).round().astype(int)
    return df


def load_student_data() -> tuple[pd.DataFrame, str]:
    """Load UCI math data, falling back to deterministic demo data."""
    if LOCAL_FILE.exists():
        return pd.read_csv(LOCAL_FILE, sep=";"), "local UCI file"

    try:
        with urlopen(UCI_ZIP_URL, timeout=20) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            with archive.open("student-mat.csv") as csv_file:
                df = pd.read_csv(csv_file, sep=";")
        df.to_csv(LOCAL_FILE, index=False)
        return df, "UCI Student Performance dataset"
    except Exception as exc:
        df = _generate_fallback_dataset()
        return df, f"fallback demo dataset ({type(exc).__name__})"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.dropna(subset=["G3"])
    cleaned["G3"] = pd.to_numeric(cleaned["G3"], errors="coerce")
    cleaned = cleaned.dropna(subset=["G3"])
    return cleaned
