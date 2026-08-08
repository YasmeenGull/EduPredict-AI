import pandas as pd

from src.config import FEATURES
from src.data_loader import _generate_fallback_dataset
from src.recommender import build_recommendations, performance_band
from src.preprocessing import build_preprocessor


def test_fallback_dataset_has_required_features():
    df = _generate_fallback_dataset(50)
    assert set(FEATURES).issubset(df.columns)
    assert "G3" in df.columns


def test_preprocessor_fits():
    df = _generate_fallback_dataset(50)
    X = df[FEATURES]
    transformed = build_preprocessor().fit_transform(X)
    assert transformed.shape[0] == 50
    assert transformed.shape[1] > 0


def test_recommendation_logic():
    assert performance_band(17) == "Excellent"
    tips = build_recommendations({"studytime": 1, "failures": 1, "absences": 10, "G1": 12, "G2": 10}, 8)
    assert len(tips) > 0
