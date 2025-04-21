#!/usr/bin/env python
"""
Script for evaluating and comparing teacher and student models.
"""
import os
import sys
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple
import hydra
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from omegaconf import DictConfig, OmegaConf

# Add the project root to the Python path if not already added
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.imdb_datamodule import IMDBDataModule
from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for the positive class
        
    Returns:
        Dictionary with metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
    except ValueError:
        roc_auc = 0.5  # Default for cases where there's only one class
    
    # Confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


def compute_calibration(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Compute calibration metrics (expected calibration error).
    
    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities for the positive class
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    # Get probabilities for positive class
    probs = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    
    # Create bins and compute ECE
    bin_indices = np.minimum(n_bins - 1, np.floor(probs * n_bins)).astype(int)
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_accs[i] = np.mean(y_true[bin_mask] == 1)
            bin_confs[i] = np.mean(probs[bin_mask])
            bin_counts[i] = np.sum(bin_mask)
    
    # Expected Calibration Error
    ece = np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / np.sum(bin_counts)
    
    return {
        "ece": float(ece),
        "bin_counts": bin_counts.tolist(),
        "bin_accs": bin_accs.tolist(),
        "bin_confs": bin_confs.tolist()
    }


def plot_decision_path_histogram(student: StudentModel, X: np.ndarray, output_dir: str) -> None:
    """Plot histogram of decision path lengths.
    
    Args:
        student: Trained student model
        X: TF-IDF features
        output_dir: Directory to save the plot to
    """
    # Compute path lengths
    path_lengths = student.compute_path_lengths(X)
    
    # Ensure path_lengths is a 1D array
    if path_lengths.ndim > 1:
        path_lengths = path_lengths.flatten()
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(path_lengths, bins=30, alpha=0.7)
    plt.xlabel('Decision Path Length')
    plt.ylabel('Number of Samples')
    plt.title('Histogram of Decision Tree Path Lengths')
    plt.grid(alpha=0.3)
    
    # Add stats to the plot
    stats_text = (
        f"Mean: {float(np.mean(path_lengths)):.2f}\n"
        f"Median: {float(np.median(path_lengths)):.2f}\n"
        f"Min: {int(np.min(path_lengths))}\n"
        f"Max: {int(np.max(path_lengths))}"
    )
    plt.annotate(
        stats_text,
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        ha='right',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8)
    )
    
    # Save plot
    plot_path = os.path.join(output_dir, "decision_path_histogram.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for evaluating models.
    
    Args:
        cfg: Configuration object
    """
    # Set random seed for reproducibility
    set_seed(cfg.train.seed)
    
    # Set up logging
    logger, console, tb_writer = setup_logging(cfg)
    logger.info("Starting model evaluation...")
    
    # Create output directories
    os.makedirs(os.path.dirname(cfg.paths.evaluation_report), exist_ok=True)
    
    # Print config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = IMDBDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    
    # Load teacher model
    logger.info(f"Loading teacher model from {cfg.paths.teacher_model}...")
    teacher = TeacherModel.load_from_checkpoint(cfg.paths.teacher_model, cfg=cfg)
    teacher.eval()
    
    # Load student model
    logger.info(f"Loading student model from {cfg.paths.student_model}...")
    student = StudentModel.load(cfg, cfg.paths.student_model)
    
    # Create TF-IDF features for test set
    logger.info("Creating TF-IDF features for test set...")
    test_tfidf = datamodule.create_test_tfidf_features()
    
    # Evaluate teacher on test set
    logger.info("Evaluating teacher model...")
    teacher_logits, test_labels = teacher.get_logits(datamodule.test_dataloader())
    teacher_logits = teacher_logits.numpy()
    test_labels = test_labels.numpy()
    
    teacher_probs = torch.nn.functional.softmax(torch.tensor(teacher_logits), dim=1).numpy()
    teacher_preds = np.argmax(teacher_probs, axis=1)
    
    # Evaluate student on test set
    logger.info("Evaluating student model...")
    student_probs = student.predict_proba(test_tfidf)
    student_preds = student.predict(test_tfidf)
    
    # Compute metrics
    teacher_metrics = compute_metrics(test_labels, teacher_preds, teacher_probs)
    student_metrics = compute_metrics(test_labels, student_preds, student_probs)
    
    # Calculate fidelity metrics on test set
    agreement = np.mean(teacher_preds == student_preds)
    
    # Gather all results
    results = {
        "teacher": {
            "metrics": teacher_metrics,
        },
        "student": {
            "metrics": student_metrics,
        },
        "fidelity": {
            "test_agreement": float(agreement),
        }
    }
    
    # Save evaluation report
    logger.info(f"Saving evaluation report to {cfg.paths.evaluation_report}...")
    with open(cfg.paths.evaluation_report, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results to console
    logger.info("\nEvaluation Results:")
    logger.info("Teacher Model:")
    for metric, value in teacher_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("\nStudent Model:")
    for metric, value in student_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info(f"\nTest Set Agreement: {agreement:.4f}")
    
    logger.info(f"\nEvaluation completed! Results saved to {cfg.paths.evaluation_report}")


if __name__ == "__main__":
    main() 