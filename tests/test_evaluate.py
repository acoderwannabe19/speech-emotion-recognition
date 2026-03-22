"""Tests for src/evaluate.py — confusion matrix, report, sweep plots."""
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI

from src.evaluate import (
    plot_confusion_matrix,
    print_classification_report,
    plot_sweep_results,
)


@pytest.fixture
def y_true_pred():
    """Simple true / predicted label vectors."""
    y_true = pd.Series(["angry", "happy", "sad", "angry", "happy", "sad"])
    y_pred = pd.Series(["angry", "happy", "angry", "angry", "sad", "sad"])
    return y_true, y_pred


class TestPlotConfusionMatrix:
    def test_returns_figure(self, y_true_pred):
        y_true, y_pred = y_true_pred
        fig = plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)

    def test_normalized_and_raw(self, y_true_pred):
        y_true, y_pred = y_true_pred
        fig1 = plot_confusion_matrix(y_true, y_pred, normalize="true")
        fig2 = plot_confusion_matrix(y_true, y_pred, normalize=None)
        assert isinstance(fig1, matplotlib.figure.Figure)
        assert isinstance(fig2, matplotlib.figure.Figure)
        matplotlib.pyplot.close("all")


class TestPrintClassificationReport:
    def test_returns_string(self, y_true_pred):
        y_true, y_pred = y_true_pred
        report = print_classification_report(y_true, y_pred)
        assert isinstance(report, str)

    def test_contains_metrics(self, y_true_pred):
        y_true, y_pred = y_true_pred
        report = print_classification_report(y_true, y_pred)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report


class TestPlotSweepResults:
    def test_returns_figure(self):
        df = pd.DataFrame({
            "run_name": ["run_1", "run_2", "run_3"],
            "model_type": ["SVM", "RF", "SVM"],
            "f1_weighted": [0.80, 0.75, 0.82],
        })
        fig = plot_sweep_results(df, metric="f1_weighted")
        assert isinstance(fig, matplotlib.figure.Figure)
        matplotlib.pyplot.close(fig)
