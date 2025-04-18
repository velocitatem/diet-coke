from typing import Dict, Any, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.base import BaseEstimator
import joblib
from omegaconf import DictConfig, OmegaConf


class StudentModel:
    """Decision Tree student model."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize the model.
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.target_type = cfg.target_type
        
        # Initialize the appropriate model type
        if self.target_type == "classification":
            self.model = DecisionTreeClassifier(
                criterion=cfg.model.student.criterion,
                max_depth=cfg.model.student.max_depth,
                min_samples_split=cfg.model.student.min_samples_split,
                min_samples_leaf=cfg.model.student.min_samples_leaf,
                max_features=cfg.model.student.max_features,
                class_weight=cfg.model.student.class_weight,
                random_state=cfg.train.seed
            )
        else:  # regression
            self.model = DecisionTreeRegressor(
                criterion="squared_error",
                max_depth=cfg.model.student.max_depth,
                min_samples_split=cfg.model.student.min_samples_split,
                min_samples_leaf=cfg.model.student.min_samples_leaf,
                max_features=cfg.model.student.max_features,
                random_state=cfg.train.seed
            )
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sample_weight: Optional[np.ndarray] = None
    ) -> "StudentModel":
        """Fit the model to the data.
        
        Args:
            X: TF-IDF features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_classes)
            sample_weight: Optional weights for samples
            
        Returns:
            Self
        """
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Args:
            X: TF-IDF features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.
        
        Args:
            X: TF-IDF features of shape (n_samples, n_features)
            
        Returns:
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        if self.target_type == "classification":
            return self.model.predict_proba(X)
        else:
            # For regression trees, return raw values reshaped to (n_samples, n_classes)
            raw_preds = self.model.predict(X)
            if raw_preds.ndim == 1:
                # Binary case: turn into [1-p, p] format
                pos_probs = raw_preds.clip(0, 1)
                return np.vstack((1 - pos_probs, pos_probs)).T
            return raw_preds
    
    def get_tree_info(self) -> Dict[str, Any]:
        """Get information about the trained tree.
        
        Returns:
            Dictionary with tree statistics
        """
        return {
            "n_nodes": self.model.tree_.node_count,
            "max_depth": self.model.tree_.max_depth,
            "n_leaves": self.model.tree_.n_leaves,
            "tree_type": self.target_type
        }
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances from the model.
        
        Returns:
            Array of feature importances
        """
        return self.model.feature_importances_
    
    def get_tree_text(self, feature_names: Optional[List[str]] = None) -> str:
        """Get a text representation of the tree.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Text representation of the tree
        """
        return export_text(
            self.model,
            feature_names=feature_names,
            show_weights=True
        )
    
    def save(self, save_path: str) -> None:
        """Save the model to disk.
        
        Args:
            save_path: Path to save the model to
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
    
    @classmethod
    def load(cls, config: DictConfig, model_path: str) -> "StudentModel":
        """Load a model from disk.
        
        Args:
            config: Configuration object
            model_path: Path to load the model from
            
        Returns:
            Loaded model
        """
        instance = cls(config)
        instance.model = joblib.load(model_path)
        return instance
    
    def compute_path_lengths(self, X: np.ndarray) -> np.ndarray:
        """Compute the decision path lengths for samples in X.
        
        Args:
            X: TF-IDF features of shape (n_samples, n_features)
            
        Returns:
            Array of path lengths for each sample
        """
        decision_path = self.model.decision_path(X)
        return decision_path.sum(axis=1) 