#!/usr/bin/env python
"""
Script for distilling knowledge from BERT to Decision Tree.
"""
import os
import sys
import json
from typing import Dict, Any
import hydra
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Add the project root to the Python path if not already added
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.imdb_datamodule import IMDBDataModule
from src.models.teacher import TeacherModel
from src.models.student import StudentModel
from src.models.distiller import Distiller
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.registry.model_registrar import ModelRegistrar


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for distilling from teacher to student.
    
    Args:
        cfg: Configuration object
    """
    # Set random seed for reproducibility
    set_seed(cfg.train.seed)
    
    # Set up logging
    logger, console, tb_writer = setup_logging(cfg)
    logger.info("Starting distillation from BERT to Decision Tree...")
    
    # Create output directories
    os.makedirs(os.path.dirname(cfg.paths.student_model), exist_ok=True)
    
    # Print config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = IMDBDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    # Load teacher model
    logger.info(f"Loading teacher model from {cfg.paths.teacher_model}...")
    
    # Check if the teacher model file exists
    if not os.path.exists(cfg.paths.teacher_model):
        logger.error(f"Teacher model not found at {cfg.paths.teacher_model}")
        
        # Try to find teacher checkpoints in outputs directory
        possible_checkpoints = []
        for root, dirs, files in os.walk('outputs'):
            for file in files:
                if file == 'teacher.ckpt' or file.endswith('.ckpt'):
                    checkpoint_path = os.path.join(root, file)
                    possible_checkpoints.append(checkpoint_path)
        
        if possible_checkpoints:
            logger.info(f"Found {len(possible_checkpoints)} possible teacher checkpoints:")
            for i, path in enumerate(possible_checkpoints):
                logger.info(f"  {i+1}. {path}")
            logger.info("Try running with: python src/distill_to_tree.py paths.teacher_model=<checkpoint_path>")
        else:
            logger.info("No teacher checkpoints found. Make sure to run train_teacher.py first.")
        
        sys.exit(1)
    
    # Determine the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load the teacher model and move it to the device
    teacher = TeacherModel.load_from_checkpoint(cfg.paths.teacher_model, cfg=cfg)
    teacher.to(device)
    teacher.eval()
    logger.info(f"Teacher model loaded and moved to {device}")
    
    # Extract TF-IDF features
    logger.info("Extracting TF-IDF features...")
    train_df, tfidf_features = datamodule.get_tfidf_data()
    
    # Initialize distiller
    logger.info("Initializing distiller...")
    distiller = Distiller(cfg, writer=tb_writer)
    
    # Perform distillation
    logger.info("Performing distillation...")
    student, distill_df = distiller.distill(
        teacher=teacher,
        dataloader=datamodule.train_dataloader(),
        tfidf_features=tfidf_features,
        texts=train_df['text'].tolist()
    )
    
    # Save student model and vectorizer
    logger.info(f"Saving student model to {cfg.paths.student_model}...")
    student.save(cfg.paths.student_model)
    
    logger.info(f"Saving TF-IDF vectorizer to {cfg.paths.vectorizer}...")
    datamodule.save_vectorizer(cfg.paths.vectorizer)
    
    # Save distillation data
    logger.info(f"Saving distillation data to {cfg.paths.distill_data}...")
    os.makedirs(os.path.dirname(cfg.paths.distill_data), exist_ok=True)
    distill_df.to_parquet(cfg.paths.distill_data)
    
    # Evaluate fidelity
    logger.info("Evaluating fidelity on training data...")
    teacher_logits, _ = teacher.get_logits(datamodule.train_dataloader())
    teacher_probs = torch.nn.functional.softmax(teacher_logits / cfg.T, dim=1).numpy()
    
    fidelity_metrics = distiller.evaluate_fidelity(
        student=student,
        teacher_outputs=teacher_probs,
        tfidf_features=tfidf_features
    )
    
    # Log fidelity metrics
    for key, value in fidelity_metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Get tree info
    tree_info = student.get_model_info()
    logger.info(f"Model info: {tree_info}")
    
    # Save metrics
    metrics = {**fidelity_metrics, **tree_info}
    metrics_file = os.path.join(os.path.dirname(cfg.paths.student_model), "distill_metrics.json")
    with open(metrics_file, "w") as f:
        # Convert any numpy values to Python types
        serializable_metrics = {k: v.item() if isinstance(v, np.number) else v for k, v in metrics.items()}
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info(f"Distillation metrics saved to {metrics_file}")
    
    # Register model in the registry
    if cfg.get("registry", {}).get("enabled", False):
        try:
            logger.info("Registering model in the registry...")
            registry_dir = cfg.registry.get("dir", os.path.join(project_root, "model_registry"))
            
            # Create model registrar
            registrar = ModelRegistrar(registry_dir)
            
            # Generate model ID if not specified
            model_id = cfg.registry.get("model_id", None)
            
            # Generate description if not specified
            model_type = cfg.model.student.get("model_type", "decision_tree")
            task_type = cfg.get("target_type", "classification")
            description = cfg.registry.get("description", 
                f"{model_type.capitalize()} model distilled from BERT for {task_type} on {cfg.data.dataset_name}"
            )
            
            # Register the model
            model_id = registrar.register_from_run(
                output_dir=cfg.output_dir,
                model_id=model_id,
                description=description,
                metrics=serializable_metrics,
                evaluation_file="distill_metrics.json"
            )
            
            if model_id:
                logger.info(f"Model registered with ID: {model_id}")
                
                # Export best models for inference if requested
                if cfg.registry.get("export_best", False):
                    inference_dir = cfg.registry.get("inference_dir", os.path.join(project_root, "inference", "models"))
                    
                    exported = registrar.register_best_models_for_inference(
                        inference_dir=inference_dir,
                        task_types=[task_type],
                        model_types=None  # Export all model types
                    )
                    
                    logger.info(f"Exported {len(exported)} best models to {inference_dir}")
            else:
                logger.error("Failed to register model in the registry")
        except Exception as e:
            logger.error(f"Error during model registration: {e}")
            logger.info("Continuing without model registration.")
    else:
        logger.info("Model registry is disabled. Skipping registration.")
    
    logger.info("Distillation completed!")


if __name__ == "__main__":
    main() 