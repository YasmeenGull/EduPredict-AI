from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")


def save_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["G3"], bins=12, kde=True, ax=axes[0], color="#4F46E5")
    axes[0].set_title("Final Grade Distribution")
    sns.scatterplot(data=df, x="studytime", y="G3", alpha=0.55, ax=axes[1], color="#0EA5E9")
    axes[1].set_title("Study Time vs Final Grade")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=160)
    plt.close(fig)


def save_prediction_plot(y_true, y_pred, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, color="#7C3AED")
    ax.plot([0, 20], [0, 20], "--", color="#EF4444", label="Perfect prediction")
    ax.set(xlabel="Actual Grade", ylabel="Predicted Grade", title="Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def save_model_comparison(metrics: dict[str, dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(metrics).T
    fig, ax = plt.subplots(figsize=(8, 4.5))
    frame["RMSE"].plot(kind="bar", ax=ax, color=["#F59E0B", "#10B981"])
    ax.set_ylabel("RMSE (lower is better)")
    ax.set_title("Model Error Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)
