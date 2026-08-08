from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import FEATURES, MODEL_DIR
from src.models import load_model
from src.recommender import build_recommendations, performance_band


def predict(profile: dict) -> tuple[float, str, list[str]]:
    missing = [feature for feature in FEATURES if feature not in profile]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    row = pd.DataFrame([profile], columns=FEATURES)
    rf = load_model("random_forest.joblib")
    nn = load_model("neural_network.joblib")
    rf_score = float(rf.predict(row)[0])
    nn_score = float(nn.predict(row)[0])
    hybrid = max(0.0, min(20.0, 0.60 * rf_score + 0.40 * nn_score))
    band = performance_band(hybrid)
    recommendations = build_recommendations(profile, hybrid)
    return hybrid, band, recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="EduPredict AI CLI prediction")
    parser.add_argument("--json", required=True, help="Path to a JSON profile file")
    args = parser.parse_args()

    profile = json.loads(open(args.json, encoding="utf-8").read())
    score, band, recommendations = predict(profile)
    print(f"Predicted final grade: {score:.2f}/20")
    print(f"Performance band: {band}")
    print("Recommendations:")
    for item in recommendations:
        print(f"- {item}")


if __name__ == "__main__":
    main()
