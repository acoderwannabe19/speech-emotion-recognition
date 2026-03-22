"""
Evaluation helpers — confusion matrix, classification report, plots.
"""

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


def plot_confusion_matrix(
    y_true, y_pred, title="Confusion Matrix", figsize=(12, 10), normalize="true"
):
    """
    Plot a confusion matrix with seaborn.
    normalize: 'true' | 'pred' | 'all' | None
    """
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlOrBr",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def print_classification_report(y_true, y_pred):
    """Print and return a classification report string."""
    report = classification_report(y_true, y_pred)
    print(report)
    return report


def plot_sweep_results(results_df: pd.DataFrame, metric: str = "f1_weighted", figsize=(14, 6)):
    """Bar chart comparing all sweep runs, coloured by model type."""
    df = results_df.sort_values(metric, ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", n_colors=df["model_type"].nunique())
    model_colors = dict(zip(df["model_type"].unique(), colors))

    ax.barh(
        df["run_name"],
        df[metric],
        color=[model_colors[m] for m in df["model_type"]],
    )
    ax.set_xlabel(metric)
    ax.set_title(f"Sweep results — {metric}")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch

    legend_handles = [Patch(color=c, label=m) for m, c in model_colors.items()]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    return fig


def load_best_model(run_id: str):
    """Load a logged sklearn model from an MLflow run."""
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)
