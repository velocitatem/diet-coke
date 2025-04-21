#!/usr/bin/env python
"""
Model Registrar: Integration with training workflows.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .model_registry import ModelRegistry
from ..models.student import StudentModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ModelRegistrar:
    """Interface for registering models during training."""
    
    def __init__(self, registry_dir: str):
        """Initialize the model registrar.
        
        Args:
            registry_dir: Path to the registry directory
        """
        self.registry = ModelRegistry(registry_dir)
    
    def register_from_run(
        self,
        output_dir: str,
        model_id: Optional[str] = None,
        description: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        evaluation_file: str = "metrics.json"
    ) -> Optional[str]:
        """Register a model from a training run output directory.
        
        Args:
            output_dir: Path to the output directory containing model artifacts
            model_id: Optional model ID (will be generated if not provided)
            description: Optional model description
            metrics: Optional metrics dictionary
            evaluation_file: Name of the evaluation metrics file
            
        Returns:
            Optional[str]: Model ID if registration was successful, None otherwise
        """
        try:
            # Load configuration
            config_path = os.path.join(output_dir, "config.yaml")
            if not os.path.exists(config_path):
                logger.warning(f"Configuration file not found at {config_path}")
                return None
            
            config = OmegaConf.load(config_path)
            
            # Get model type and task type from config
            model_type = config.get("model", {}).get("student", {}).get("model_type", "decision_tree")
            task_type = config.get("target_type", "classification")
            
            # Get model and vectorizer paths
            model_path = os.path.join(output_dir, "artifacts", "student.pkl")
            vectorizer_path = os.path.join(output_dir, "artifacts", "vectorizer.pkl")
            
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                logger.warning(f"Model or vectorizer files not found in {output_dir}")
                return None
            
            # Load metrics if not provided
            if metrics is None:
                metrics_path = os.path.join(output_dir, "artifacts", evaluation_file)
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                else:
                    logger.warning(f"Metrics file not found at {metrics_path}")
                    metrics = {}
            
            # Generate description if not provided
            if description is None:
                description = f"{model_type.capitalize()} model for {task_type}"
            
            # Register the model
            return self.registry.register_model(
                model_id=model_id,
                model_type=model_type,
                task_type=task_type,
                description=description,
                config=config,
                model_path=model_path,
                vectorizer_path=vectorizer_path,
                metrics=metrics,
                source_dir=output_dir,
                copy_artifacts=True
            )
        except Exception as e:
            logger.error(f"Error registering model from run: {e}")
            return None
    
    def _load_metrics(
        self, 
        output_dir: str, 
        evaluation_file: Optional[str] = None
    ) -> Dict[str, float]:
        """Load metrics from evaluation file.
        
        Args:
            output_dir: Directory containing the training outputs
            evaluation_file: Path to evaluation results (relative to output_dir)
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        metrics = {}
        
        # Try to find evaluation file
        if evaluation_file is None:
            candidates = [
                "evaluation_report.json",
                "eval_results.json",
                "metrics.json",
                "results.json"
            ]
            
            for candidate in candidates:
                path = os.path.join(output_dir, candidate)
                if os.path.exists(path):
                    evaluation_file = candidate
                    break
        
        # Load metrics from file
        if evaluation_file:
            eval_path = os.path.join(output_dir, evaluation_file)
            if os.path.exists(eval_path):
                try:
                    with open(eval_path, "r") as f:
                        metrics = json.load(f)
                    logger.info(f"Loaded metrics from {eval_path}")
                except Exception as e:
                    logger.error(f"Error loading metrics from {eval_path}: {e}")
        
        # Extract scalar metrics (filter out non-scalar values)
        scalar_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                scalar_metrics[k] = v
            elif isinstance(v, dict) and "value" in v and isinstance(v["value"], (int, float)) and not np.isnan(v["value"]):
                scalar_metrics[k] = v["value"]
        
        return scalar_metrics
    
    def register_best_models_for_inference(
        self, 
        inference_dir: str,
        task_types: Optional[List[str]] = None,
        model_types: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Export best models to the inference directory.
        
        Args:
            inference_dir: Directory to export models to
            task_types: List of task types to export (default: ["classification"])
            model_types: List of model types to export (default: all available)
            
        Returns:
            Dict[str, str]: Mapping of model types to exported model paths
        """
        # Set defaults
        if task_types is None:
            task_types = ["classification"]
        
        # Create inference directory
        os.makedirs(inference_dir, exist_ok=True)
        
        # Dictionary to track exported models
        exported_models = {}
        
        for task_type in task_types:
            # Get all available model types for this task
            if model_types is None:
                available_models = self.registry.list_models(task_type=task_type)
                task_model_types = available_models["model_type"].unique().tolist()
            else:
                task_model_types = model_types
            
            # Create task directory
            task_dir = os.path.join(inference_dir, task_type)
            os.makedirs(task_dir, exist_ok=True)
            
            # Export each model type
            for model_type in task_model_types:
                best_model = self.registry.get_best_model(task_type, model_type)
                
                if best_model:
                    model_id = best_model["model_id"]
                    model_export_dir = os.path.join(task_dir, model_type)
                    
                    success = self.registry.export_model(
                        model_id=model_id,
                        export_dir=model_export_dir,
                        include_metrics=True
                    )
                    
                    if success:
                        exported_models[f"{task_type}_{model_type}"] = model_export_dir
                        logger.info(f"Exported best {model_type} model for {task_type} to {model_export_dir}")
        
        # Create a registry summary file
        summary = {
            "exported_models": exported_models,
            "model_info": {
                model_key: {
                    "task_type": key.split('_')[0],
                    "model_type": key.split('_')[1],
                    "path": path
                }
                for key, path in exported_models.items()
            }
        }
        
        with open(os.path.join(inference_dir, "registry_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        return exported_models 