"""
Model Evaluation and Error Diagnostics Module.
Provides standard and multi-class classification metrics:
- Accuracy, Macro & Weighted Precision, Recall, F1-Score
- Confusion Matrix generation and visualization
- Training history curves (Loss and Accuracy)
- False positive / False negative misclassification analysis
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluator:
    """Evaluation suite for benchmark and hybrid IDS models."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute standard multi-class classification metrics.
        
        Args:
            y_true: Ground truth target labels.
            y_pred: Model predicted labels.
            y_prob: Optional predicted probability distribution.
            
        Returns:
            Dictionary containing accuracy, precision, recall, and f1 scores.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        acc = float(accuracy_score(y_true, y_pred))
        p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        p_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        r_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        metrics = {
            "accuracy": round(acc, 4),
            "precision_macro": round(p_macro, 4),
            "recall_macro": round(r_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(p_weighted, 4),
            "recall_weighted": round(r_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "total_samples": int(len(y_true)),
        }
        return metrics

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate detailed per-class classification report as a DataFrame.
        """
        target_names = target_names or self.class_names
        labels = list(range(len(target_names))) if target_names else None

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).transpose()
        return report_df

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (16, 14),
        normalize: bool = False,
    ) -> plt.Figure:
        """
        Generate and save confusion matrix heatmap.
        """
        target_names = target_names or self.class_names
        num_classes = len(target_names) if target_names else len(np.unique(y_true))
        labels = list(range(num_classes))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if normalize:
            cm_sum = cm.sum(axis=1)[:, np.newaxis]
            cm_display = np.divide(cm.astype("float"), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
            fmt = ".2f"
        else:
            cm_display = cm
            fmt = "d"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=target_names if target_names else labels,
            yticklabels=target_names if target_names else labels,
            ax=ax,
            cbar_kws={"label": "Normalized Fraction" if normalize else "Sample Count"},
        )
        ax.set_title("Confusion Matrix Heatmap", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
        ax.set_ylabel("True Label", fontsize=12, labelpad=10)
        plt.xticks(rotation=90, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")

        return fig

    def analyze_misclassifications(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Extract the most frequent error pairs (True Label -> Predicted Label).
        """
        target_names = target_names or self.class_names
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        errors_mask = y_true != y_pred
        error_true = y_true[errors_mask]
        error_pred = y_pred[errors_mask]

        error_records = []
        for t, p in zip(error_true, error_pred):
            t_name = target_names[t] if target_names and 0 <= t < len(target_names) else str(t)
            p_name = target_names[p] if target_names and 0 <= p < len(target_names) else str(p)
            error_records.append((t_name, p_name))

        if not error_records:
            return pd.DataFrame(columns=["True Class", "Predicted Class", "Error Count", "Error Percentage"])

        df_errors = pd.DataFrame(error_records, columns=["True Class", "Predicted Class"])
        summary = (
            df_errors.groupby(["True Class", "Predicted Class"])
            .size()
            .reset_index(name="Error Count")
            .sort_values(by="Error Count", ascending=False)
        )
        total_errors = len(error_records)
        summary["Error Percentage"] = (summary["Error Count"] / total_errors * 100).round(2)
        return summary.head(top_n)
