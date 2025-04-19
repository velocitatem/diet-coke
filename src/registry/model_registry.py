#!/usr/bin/env python
"""
Model Registry for tracking and comparing distilled models.
"""

import os
import json
import shutil
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf, DictConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for tracking and managing distilled models."""
    
    def __init__(self, registry_dir: str):
        """Initialize the model registry.
        
        Args:
            registry_dir: Path to the registry directory
        """
        self.registry_dir = registry_dir
        self.models_dir = os.path.join(registry_dir, "models")
        self.index_file = os.path.join(registry_dir, "model_index.json")
        self.metrics_file = os.path.join(registry_dir, "model_metrics.csv")
        self.best_models_file = os.path.join(registry_dir, "best_models.json")
        
        # Create registry directories if they don't exist
        os.makedirs(self.registry_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize or load the model index
        if os.path.exists(self.index_file):
            with open(self.index_file, "r") as f:
                self.model_index = json.load(f)
        else:
            self.model_index = {}
            self._save_index()
        
        # Initialize or load the metrics dataframe
        if os.path.exists(self.metrics_file) and os.path.getsize(self.metrics_file) > 0:
            try:
                self.metrics_df = pd.read_csv(self.metrics_file)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                logger.warning(f"Could not read metrics file {self.metrics_file}. Creating a new one.")
                self.metrics_df = pd.DataFrame(columns=["model_id", "model_type", "task_type", "accuracy", "fidelity"])
                self._save_metrics()
        else:
            # Create a new metrics file with appropriate column headers
            self.metrics_df = pd.DataFrame(columns=["model_id", "model_type", "task_type", "accuracy", "fidelity"])
            self._save_metrics()
            
        # Initialize or load the best models index
        if os.path.exists(self.best_models_file):
            with open(self.best_models_file, "r") as f:
                self.best_models = json.load(f)
        else:
            self.best_models = {
                "classification": {},
                "regression": {}
            }
            self._save_best_models()
    
    def _save_index(self):
        """Save the model index to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self.model_index, f, indent=2)
    
    def _save_metrics(self):
        """Save the metrics dataframe to disk."""
        self.metrics_df.to_csv(self.metrics_file, index=False)
    
    def _save_best_models(self):
        """Save the best models index to disk."""
        with open(self.best_models_file, "w") as f:
            json.dump(self.best_models, f, indent=2)
    
    def register_model(
        self,
        model_id: str,
        model_type: str,
        task_type: str,
        description: str,
        config: DictConfig,
        model_path: str,
        vectorizer_path: str,
        metrics: Dict[str, float],
        source_dir: Optional[str] = None,
        copy_artifacts: bool = True
    ) -> str:
        """Register a model in the registry.
        
        Args:
            model_id: Unique identifier for the model (can be auto-generated timestamp if None)
            model_type: Type of model (e.g., decision_tree, random_forest, etc.)
            task_type: Type of task (classification or regression)
            description: Description of the model
            config: Model configuration
            model_path: Path to the model file
            vectorizer_path: Path to the vectorizer file
            metrics: Performance metrics of the model
            source_dir: Source directory containing all artifacts
            copy_artifacts: Whether to copy the model artifacts to the registry
            
        Returns:
            str: Model ID
        """
        # Generate model ID if not provided
        if not model_id:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_type}_{timestamp}"
        
        # Create model directory in registry
        model_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Resolve paths
        if source_dir:
            model_path = os.path.join(source_dir, os.path.basename(model_path))
            vectorizer_path = os.path.join(source_dir, os.path.basename(vectorizer_path))
            
        # Copy artifacts if requested
        registry_model_path = model_path
        registry_vectorizer_path = vectorizer_path
        
        if copy_artifacts:
            registry_model_path = os.path.join(model_dir, os.path.basename(model_path))
            registry_vectorizer_path = os.path.join(model_dir, os.path.basename(vectorizer_path))
            
            try:
                shutil.copy2(model_path, registry_model_path)
                shutil.copy2(vectorizer_path, registry_vectorizer_path)
                
                # Save configuration
                config_path = os.path.join(model_dir, "config.yaml")
                OmegaConf.save(config, config_path)
                
                logger.info(f"Copied model artifacts to {model_dir}")
            except Exception as e:
                logger.error(f"Error copying artifacts: {e}")
                return None
        
        # Create model entry
        model_entry = {
            "model_id": model_id,
            "model_type": model_type,
            "task_type": task_type,
            "description": description,
            "model_path": registry_model_path,
            "vectorizer_path": registry_vectorizer_path,
            "config_path": os.path.join(model_dir, "config.yaml") if copy_artifacts else None,
            "metrics": metrics,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Add to the model index
        self.model_index[model_id] = model_entry
        self._save_index()
        
        # Add to the metrics dataframe
        metrics_row = {"model_id": model_id, "model_type": model_type, "task_type": task_type}
        metrics_row.update(metrics)
        
        self.metrics_df = pd.concat([
            self.metrics_df, 
            pd.DataFrame([metrics_row])
        ], ignore_index=True)
        
        self._save_metrics()
        
        # Update best models if this model is better
        self._update_best_models(model_id, task_type, model_type, metrics)
        
        logger.info(f"Registered model {model_id} in the registry")
        return model_id
    
    def _update_best_models(
        self, 
        model_id: str, 
        task_type: str, 
        model_type: str, 
        metrics: Dict[str, float]
    ):
        """Update the best models index based on metrics.
        
        Args:
            model_id: ID of the model
            task_type: Type of task (classification or regression)
            model_type: Type of model
            metrics: Performance metrics of the model
        """
        if task_type not in self.best_models:
            self.best_models[task_type] = {}
        
        # Define metric to optimize based on task type
        if task_type == "classification":
            primary_metric = "accuracy"
            secondary_metric = "macro_f1"
        else:  # regression
            primary_metric = "r2"
            secondary_metric = "rmse"  # Lower is better for RMSE
        
        # Check if the current model type exists in best models
        if model_type not in self.best_models[task_type]:
            self.best_models[task_type][model_type] = model_id
        else:
            # Get the current best model
            current_best_id = self.best_models[task_type][model_type]
            current_best_metrics = self.model_index[current_best_id]["metrics"]
            
            # Compare metrics
            if primary_metric in metrics and primary_metric in current_best_metrics:
                if (task_type == "regression" and secondary_metric == "rmse"):
                    # For RMSE, lower is better
                    if metrics[primary_metric] > current_best_metrics[primary_metric] or (
                        metrics[primary_metric] == current_best_metrics[primary_metric] and
                        metrics.get(secondary_metric, float('inf')) < current_best_metrics.get(secondary_metric, float('inf'))
                    ):
                        self.best_models[task_type][model_type] = model_id
                else:
                    # For other metrics, higher is better
                    if metrics[primary_metric] > current_best_metrics[primary_metric] or (
                        metrics[primary_metric] == current_best_metrics[primary_metric] and
                        metrics.get(secondary_metric, 0) > current_best_metrics.get(secondary_metric, 0)
                    ):
                        self.best_models[task_type][model_type] = model_id
        
        # Also track the overall best model for the task type
        best_key = f"best_overall"
        if best_key not in self.best_models[task_type]:
            self.best_models[task_type][best_key] = model_id
        else:
            # Get the current overall best model
            current_best_id = self.best_models[task_type][best_key]
            current_best_metrics = self.model_index[current_best_id]["metrics"]
            
            # Compare metrics
            if primary_metric in metrics and primary_metric in current_best_metrics:
                if (task_type == "regression" and secondary_metric == "rmse"):
                    # For RMSE, lower is better
                    if metrics[primary_metric] > current_best_metrics[primary_metric] or (
                        metrics[primary_metric] == current_best_metrics[primary_metric] and
                        metrics.get(secondary_metric, float('inf')) < current_best_metrics.get(secondary_metric, float('inf'))
                    ):
                        self.best_models[task_type][best_key] = model_id
                else:
                    # For other metrics, higher is better
                    if metrics[primary_metric] > current_best_metrics[primary_metric] or (
                        metrics[primary_metric] == current_best_metrics[primary_metric] and
                        metrics.get(secondary_metric, 0) > current_best_metrics.get(secondary_metric, 0)
                    ):
                        self.best_models[task_type][best_key] = model_id
        
        self._save_best_models()
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get a model by ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dict[str, Any]: Model entry or None if not found
        """
        return self.model_index.get(model_id)
    
    def get_best_model(self, task_type: str, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get the best model for a task and model type.
        
        Args:
            task_type: Type of task (classification or regression)
            model_type: Type of model (if None, returns the overall best model)
            
        Returns:
            Dict[str, Any]: Best model entry or None if not found
        """
        if task_type not in self.best_models:
            return None
        
        if model_type is None:
            model_id = self.best_models[task_type].get("best_overall")
        else:
            model_id = self.best_models[task_type].get(model_type)
        
        if model_id:
            return self.get_model(model_id)
        return None
    
    def list_models(
        self, 
        task_type: Optional[str] = None, 
        model_type: Optional[str] = None
    ) -> pd.DataFrame:
        """List models in the registry, optionally filtered by task and model type.
        
        Args:
            task_type: Type of task to filter by
            model_type: Type of model to filter by
            
        Returns:
            pd.DataFrame: Filtered metrics dataframe
        """
        df = self.metrics_df.copy()
        
        if task_type:
            df = df[df["task_type"] == task_type]
        
        if model_type:
            df = df[df["model_type"] == model_type]
        
        return df
    
    def compare_models(
        self, 
        model_ids: List[str],
        metrics: Optional[List[str]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False
    ) -> pd.DataFrame:
        """Compare specified models based on their metrics.
        
        Args:
            model_ids: List of model IDs to compare
            metrics: List of metrics to include (if None, includes all)
            sort_by: Metric to sort by
            ascending: Whether to sort in ascending order
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        # Filter metrics dataframe by model IDs
        df = self.metrics_df[self.metrics_df["model_id"].isin(model_ids)].copy()
        
        # Select specific metrics if provided
        if metrics:
            cols = ["model_id", "model_type", "task_type"] + metrics
            df = df[cols]
        
        # Sort by specified metric
        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
        
        return df
    
    def plot_model_comparison(
        self, 
        model_ids: List[str],
        metric: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ):
        """Create a bar chart comparing models on a specific metric.
        
        Args:
            model_ids: List of model IDs to compare
            metric: Metric to compare
            output_path: Path to save the plot (if None, displays the plot)
            title: Title for the plot
        """
        # Get comparison dataframe
        df = self.compare_models(model_ids, metrics=[metric])
        
        if df.empty or metric not in df.columns:
            logger.error(f"No data available for metric {metric}")
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create labels with model type and ID
        labels = [f"{row['model_type']} ({row['model_id']})" for _, row in df.iterrows()]
        
        # Plot bars
        sns.barplot(x=labels, y=df[metric])
        
        # Add labels and title
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(title or f"Model Comparison by {metric}")
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        
        # Save or display the plot
        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved comparison plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_model(
        self, 
        model_id: str, 
        export_dir: str, 
        include_metrics: bool = True
    ) -> bool:
        """Export a model to a specified directory.
        
        Args:
            model_id: ID of the model to export
            export_dir: Directory to export the model to
            include_metrics: Whether to include metrics in the export
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_entry = self.get_model(model_id)
        if not model_entry:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        try:
            # Create export directory
            os.makedirs(export_dir, exist_ok=True)
            
            # Copy model artifacts
            shutil.copy2(model_entry["model_path"], os.path.join(export_dir, os.path.basename(model_entry["model_path"])))
            shutil.copy2(model_entry["vectorizer_path"], os.path.join(export_dir, os.path.basename(model_entry["vectorizer_path"])))
            
            # Copy config if available
            if model_entry["config_path"] and os.path.exists(model_entry["config_path"]):
                shutil.copy2(model_entry["config_path"], os.path.join(export_dir, "config.yaml"))
            
            # Create model info file
            model_info = {k: v for k, v in model_entry.items() if k != "metrics" or include_metrics}
            
            with open(os.path.join(export_dir, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Exported model {model_id} to {export_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model {model_id}: {e}")
            return False
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model from the registry.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        model_entry = self.get_model(model_id)
        if not model_entry:
            logger.error(f"Model {model_id} not found in registry")
            return False
        
        try:
            # Remove from indices
            del self.model_index[model_id]
            self._save_index()
            
            # Remove from metrics dataframe
            self.metrics_df = self.metrics_df[self.metrics_df["model_id"] != model_id]
            self._save_metrics()
            
            # Update best models if necessary
            for task_type in self.best_models:
                for model_type, best_id in list(self.best_models[task_type].items()):
                    if best_id == model_id:
                        # Find the next best model of this type
                        candidates = self.metrics_df[
                            (self.metrics_df["task_type"] == task_type) & 
                            (self.metrics_df["model_type"] == model_type)
                        ]
                        
                        if len(candidates) > 0:
                            # Sort candidates by appropriate metric
                            if task_type == "classification":
                                candidates = candidates.sort_values(by="accuracy", ascending=False)
                            else:  # regression
                                candidates = candidates.sort_values(by="r2", ascending=False)
                            
                            if len(candidates) > 0:
                                self.best_models[task_type][model_type] = candidates.iloc[0]["model_id"]
                            else:
                                del self.best_models[task_type][model_type]
                        else:
                            del self.best_models[task_type][model_type]
            
            self._save_best_models()
            
            # Remove model directory
            model_dir = os.path.join(self.models_dir, model_id)
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
            
            logger.info(f"Deleted model {model_id} from registry")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False