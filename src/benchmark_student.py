#!/usr/bin/env python
"""
Script for benchmarking different student model configurations.
"""
import os
import sys
import json
import time
from typing import Dict, Any, List
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch

# Add the project root to the Python path if not already added
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.data.imdb_datamodule import IMDBDataModule
from src.models.teacher import TeacherModel
from src.models.student import StudentModel


def create_model_configs() -> List[Dict[str, Any]]:
    """Create different model configurations to benchmark.
    
    Returns:
        List of model configurations
    """
    configs = []
    
    # Decision Tree configurations
    tree_configs = [
        {
            "model_type": "decision_tree",
            "criterion": "gini",
            "max_depth": depth,
            "min_samples_split": split,
            "min_samples_leaf": leaf,
            "max_features": features,
            "class_weight": weight
        }
        for depth in [4, 8, 12]
        for split in [10, 20, 30]
        for leaf in [5, 10, 15]
        for features in ["sqrt", "log2"]
        for weight in [None, "balanced"]
    ]
    configs.extend(tree_configs)
    
    # Random Forest configurations
    forest_configs = [
        {
            "model_type": "random_forest",
            "n_estimators": n_estimators,
            "max_depth": depth,
            "min_samples_split": split,
            "min_samples_leaf": leaf,
            "max_features": features,
            "class_weight": weight
        }
        for n_estimators in [50, 100, 200]
        for depth in [4, 8, 12]
        for split in [10, 20, 30]
        for leaf in [5, 10, 15]
        for features in ["sqrt", "log2"]
        for weight in [None, "balanced"]
    ]
    configs.extend(forest_configs)
    
    # Linear model configurations
    linear_configs = [
        {
            "model_type": "linear",
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "max_iter": max_iter
        }
        for C in [0.1, 1.0, 10.0]
        for penalty in ["l1", "l2"]
        for solver in ["liblinear", "saga"]
        for max_iter in [1000, 2000]
    ]
    configs.extend(linear_configs)
    
    return configs


def evaluate_model(
    student: StudentModel,
    test_tfidf: np.ndarray,
    test_labels: np.ndarray,
    teacher_preds: np.ndarray
) -> Dict[str, float]:
    """Evaluate a student model.
    
    Args:
        student: Trained student model
        test_tfidf: TF-IDF features for test set
        test_labels: True labels for test set
        teacher_preds: Teacher model predictions
        
    Returns:
        Dictionary of metrics
    """
    # Get predictions
    student_preds = student.predict(test_tfidf)
    student_probs = student.predict_proba(test_tfidf)
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(test_labels, student_preds),
        "precision": precision_score(test_labels, student_preds),
        "recall": recall_score(test_labels, student_preds),
        "f1": f1_score(test_labels, student_preds),
        "roc_auc": roc_auc_score(test_labels, student_probs[:, 1]),
        "agreement": np.mean(teacher_preds == student_preds),
        "true_negatives": np.sum((student_preds == 0) & (test_labels == 0)),
        "false_positives": np.sum((student_preds == 1) & (test_labels == 0)),
        "false_negatives": np.sum((student_preds == 0) & (test_labels == 1)),
        "true_positives": np.sum((student_preds == 1) & (test_labels == 1))
    }
    
    return metrics


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for benchmarking student models.
    
    Args:
        cfg: Configuration object
    """
    # Set random seed for reproducibility
    set_seed(cfg.train.seed)
    
    # Set up logging
    logger, console, tb_writer = setup_logging(cfg)
    logger.info("Starting student model benchmarking...")
    
    # Create output directories
    benchmark_dir = os.path.dirname(cfg.paths.benchmark.report)
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = IMDBDataModule(cfg)
    datamodule.prepare_data()
    
    # Check if vectorizer exists, if not create it from training data
    if not os.path.exists(cfg.paths.vectorizer):
        logger.info("Vectorizer not found. Creating from training data...")
        datamodule.setup(stage="fit")
        datamodule.save_vectorizer(cfg.paths.vectorizer)
        logger.info(f"Vectorizer saved to {cfg.paths.vectorizer}")
    
    # Setup for test
    datamodule.setup(stage="test")
    
    # Load teacher model
    logger.info(f"Loading teacher model from {cfg.paths.teacher_model}...")
    teacher = TeacherModel.load_from_checkpoint(cfg.paths.teacher_model, cfg=cfg)
    teacher.eval()
    
    # Get teacher predictions
    logger.info("Getting teacher predictions...")
    teacher_logits, test_labels = teacher.get_logits(datamodule.test_dataloader())
    teacher_logits = teacher_logits.numpy()
    test_labels = test_labels.numpy()
    teacher_probs = torch.nn.functional.softmax(torch.tensor(teacher_logits), dim=1).numpy()
    teacher_preds = np.argmax(teacher_probs, axis=1)
    
    # Create TF-IDF features
    logger.info("Creating TF-IDF features...")
    test_tfidf = datamodule.create_test_tfidf_features()
    
    # Get model configurations
    model_configs = create_model_configs()
    logger.info(f"Created {len(model_configs)} model configurations to test")
    
    # Benchmark models
    results = []
    for i, model_config in enumerate(model_configs):
        logger.info(f"Testing configuration {i+1}/{len(model_configs)}")
        logger.info(f"Configuration: {model_config}")
        
        # Create and train student model
        start_time = time.time()
        student = StudentModel(cfg, model_config)
        student.fit(test_tfidf, teacher_preds)
        training_time = time.time() - start_time
        
        # Evaluate model
        metrics = evaluate_model(student, test_tfidf, test_labels, teacher_preds)
        metrics["training_time"] = training_time
        metrics["config"] = model_config
        
        results.append(metrics)
        
        # Log progress
        logger.info(f"Results: {metrics}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    logger.info(f"Saving benchmark results to {cfg.paths.benchmark.report}...")
    df.to_csv(cfg.paths.benchmark.report, index=False)
    
    # Print summary
    logger.info("\nBenchmark Summary:")
    logger.info(f"Best accuracy: {df['accuracy'].max():.4f}")
    logger.info(f"Best F1: {df['f1'].max():.4f}")
    logger.info(f"Best agreement: {df['agreement'].max():.4f}")
    
    # Save best configurations
    best_configs = {
        "best_accuracy": df.loc[df['accuracy'].idxmax()]['config'],
        "best_f1": df.loc[df['f1'].idxmax()]['config'],
        "best_agreement": df.loc[df['agreement'].idxmax()]['config']
    }
    
    with open(cfg.paths.benchmark.best_configs, 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    logger.info("Benchmarking completed!")


if __name__ == "__main__":
    main() 