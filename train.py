from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ARTIFACT_DIR, FEATURES, RANDOM_STATE, TARGET, TEST_SIZE
from src.data_loader import clean_data, load_student_data
from src.models import build_neural_network, build_random_forest, metrics, save_model
from src.preprocessing import validate_features
from src.visualization import save_eda_plots, save_model_comparison, save_prediction_plot


def main() -> None:
    raw_df, source = load_student_data()
    df = clean_data(raw_df)
    validate_features(df)

    df["G3"] = df["G3"].clip(0, 20)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rf = build_random_forest()
    nn = build_neural_network()
    rf.fit(X_train, y_train)
    nn.fit(X_train, y_train)

    rf_pred = rf.predict(X_test).clip(0, 20)
    nn_pred = nn.predict(X_test).clip(0, 20)
    hybrid_pred = (0.60 * rf_pred + 0.40 * nn_pred).clip(0, 20)

    all_metrics = {
        "Random Forest": metrics(y_test, rf_pred),
        "Neural Network": metrics(y_test, nn_pred),
        "Hybrid": metrics(y_test, hybrid_pred),
    }

    save_model(rf, "random_forest.joblib")
    save_model(nn, "neural_network.joblib")
    pd.DataFrame({"actual": y_test, "predicted": hybrid_pred}).to_csv(
        ARTIFACT_DIR / "test_predictions.csv", index=False
    )
    save_eda_plots(df, ARTIFACT_DIR)
    save_prediction_plot(y_test, hybrid_pred, ARTIFACT_DIR)
    save_model_comparison(all_metrics, ARTIFACT_DIR)

    metadata = {
        "dataset_source": source,
        "rows_after_cleaning": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "test_size": TEST_SIZE,
        "metrics": all_metrics,
        "hybrid_weights": {"random_forest": 0.60, "neural_network": 0.40},
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2))

    print("\nTraining complete.")
    print(f"Dataset: {source}")
    print(f"Rows: {len(df)}")
    print(json.dumps(all_metrics, indent=2))
    print("Models saved in models/ and charts in artifacts/.")


if __name__ == "__main__":
    main()
