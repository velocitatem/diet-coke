from typing import Dict, Any, List, Optional, Tuple, Union
import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator
import joblib
from omegaconf import DictConfig, OmegaConf


class BaseInterpretableModel:
    """Base class for interpretable models."""
    
    def __init__(self, cfg: DictConfig, target_type: str):
        """Initialize the base model.
        
        Args:
            cfg: Configuration object
            target_type: Type of task ('classification' or 'regression')
        """
        self.cfg = cfg
        self.target_type = target_type
        self.model = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "BaseInterpretableModel":
        """Fit the model to the data."""
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples in X."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, create a simple approximation
            raw_preds = self.model.predict(X)
            if raw_preds.ndim == 1:
                # Binary case: turn into [1-p, p] format
                pos_probs = raw_preds.clip(0, 1)
                return np.vstack((1 - pos_probs, pos_probs)).T
            return raw_preds
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances from the model."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            coefs = self.model.coef_
            if coefs.ndim > 1:
                return np.mean(np.abs(coefs), axis=0)
            return np.abs(coefs)
        else:
            # Return a default value for models without feature importances
            return np.array([])


class DecisionTreeModel(BaseInterpretableModel):
    """Decision Tree model implementation."""
    
    def __init__(self, cfg: DictConfig, target_type: str):
        super().__init__(cfg, target_type)
        
        if target_type == "classification":
            self.model = DecisionTreeClassifier(
                criterion=cfg.criterion,
                max_depth=cfg.max_depth,
                min_samples_split=cfg.min_samples_split,
                min_samples_leaf=cfg.min_samples_leaf,
                max_features=cfg.max_features,
                class_weight=cfg.class_weight,
                random_state=cfg.train.seed
            )
        else:  # regression
            self.model = DecisionTreeRegressor(
                criterion="squared_error",
                max_depth=cfg.max_depth,
                min_samples_split=cfg.min_samples_split,
                min_samples_leaf=cfg.min_samples_leaf,
                max_features=cfg.max_features,
                random_state=cfg.train.seed
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained tree."""
        return {
            "n_nodes": self.model.tree_.node_count,
            "max_depth": self.model.tree_.max_depth,
            "n_leaves": self.model.tree_.n_leaves,
            "model_type": "decision_tree",
            "task_type": self.target_type
        }
    
    def get_tree_text(self, feature_names: Optional[List[str]] = None) -> str:
        """Get a text representation of the tree."""
        return export_text(
            self.model,
            feature_names=feature_names,
            show_weights=True
        )
    
    def compute_path_lengths(self, X: np.ndarray) -> np.ndarray:
        """Compute the decision path lengths for samples in X."""
        decision_path = self.model.decision_path(X)
        return decision_path.sum(axis=1)


class RandomForestModel(BaseInterpretableModel):
    """Random Forest model implementation."""
    
    def __init__(self, cfg: DictConfig, target_type: str):
        super().__init__(cfg, target_type)
        
        if target_type == "classification":
            self.model = RandomForestClassifier(
                n_estimators=cfg.n_estimators,
                criterion=cfg.criterion,
                max_depth=cfg.max_depth,
                min_samples_split=cfg.min_samples_split,
                min_samples_leaf=cfg.min_samples_leaf,
                max_features=cfg.max_features,
                class_weight=cfg.class_weight,
                random_state=cfg.train.seed
            )
        else:  # regression
            self.model = RandomForestRegressor(
                n_estimators=cfg.n_estimators,
                criterion="squared_error",
                max_depth=cfg.max_depth,
                min_samples_split=cfg.min_samples_split,
                min_samples_leaf=cfg.min_samples_leaf,
                max_features=cfg.max_features,
                random_state=cfg.train.seed
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained random forest."""
        return {
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "model_type": "random_forest",
            "task_type": self.target_type
        }


class LinearModel(BaseInterpretableModel):
    """Linear model implementation (Logistic Regression for classification, Ridge for regression)."""
    
    def __init__(self, cfg: DictConfig, target_type: str):
        super().__init__(cfg, target_type)
        
        if target_type == "classification":
            self.model = LogisticRegression(
                C=cfg.C,
                penalty=cfg.penalty,
                solver=cfg.solver,
                max_iter=cfg.max_iter,
                class_weight=cfg.class_weight,
                random_state=cfg.train.seed
            )
        else:  # regression
            self.model = Ridge(
                alpha=cfg.alpha,
                max_iter=cfg.max_iter,
                solver=cfg.solver,
                random_state=cfg.train.seed
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained linear model."""
        if self.target_type == "classification":
            return {
                "model_type": "logistic_regression",
                "C": self.model.C,
                "penalty": self.model.penalty,
                "solver": self.model.solver,
                "task_type": self.target_type
            }
        else:
            return {
                "model_type": "ridge_regression",
                "alpha": self.model.alpha,
                "solver": self.model.solver,
                "task_type": self.target_type
            }


class SVMModel(BaseInterpretableModel):
    """SVM model implementation."""
    
    def __init__(self, cfg: DictConfig, target_type: str):
        super().__init__(cfg, target_type)
        
        if target_type == "classification":
            self.model = SVC(
                C=cfg.C,
                kernel=cfg.kernel,
                degree=cfg.degree if cfg.kernel == 'poly' else 3,
                gamma=cfg.gamma,
                probability=True,  # Required for predict_proba
                class_weight=cfg.class_weight,
                random_state=cfg.train.seed
            )
        else:  # regression
            self.model = SVR(
                C=cfg.C,
                kernel=cfg.kernel,
                degree=cfg.degree if cfg.kernel == 'poly' else 3,
                gamma=cfg.gamma
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained SVM model."""
        return {
            "model_type": "svm",
            "C": self.model.C,
            "kernel": self.model.kernel,
            "task_type": self.target_type
        }


class NaiveBayesModel(BaseInterpretableModel):
    """Naive Bayes model implementation (classification only)."""
    
    def __init__(self, cfg: DictConfig, target_type: str):
        super().__init__(cfg, target_type)
        
        if target_type != "classification":
            raise ValueError("Naive Bayes is only available for classification tasks")
        
        self.model = GaussianNB(
            var_smoothing=cfg.var_smoothing
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained Naive Bayes model."""
        return {
            "model_type": "naive_bayes",
            "var_smoothing": self.model.var_smoothing,
            "task_type": self.target_type
        }


def create_model(model_type: str, cfg: DictConfig, target_type: str) -> Any:
    """Create a model based on the configuration.
    
    Args:
        model_type: Type of model to create
        cfg: Configuration object
        target_type: Type of target variable
        
    Returns:
        Created model
    """
    model_classes = {
        "decision_tree": DecisionTreeModel,
        "random_forest": RandomForestModel,
        "linear": LinearModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Create a new config with only the relevant parameters
    model_cfg = OmegaConf.create({})
    
    if model_type == "decision_tree":
        model_cfg.criterion = cfg.criterion
        model_cfg.max_depth = cfg.max_depth
        model_cfg.min_samples_split = cfg.min_samples_split
        model_cfg.min_samples_leaf = cfg.min_samples_leaf
        model_cfg.max_features = cfg.max_features
        model_cfg.class_weight = cfg.class_weight
    elif model_type == "random_forest":
        model_cfg.n_estimators = cfg.n_estimators
        model_cfg.max_depth = cfg.max_depth
        model_cfg.min_samples_split = cfg.min_samples_split
        model_cfg.min_samples_leaf = cfg.min_samples_leaf
        model_cfg.max_features = cfg.max_features
        model_cfg.class_weight = cfg.class_weight
    elif model_type == "linear":
        model_cfg.C = cfg.C
        model_cfg.penalty = cfg.penalty
        model_cfg.solver = cfg.solver
        model_cfg.max_iter = cfg.max_iter
        
    return model_classes[model_type](model_cfg, target_type)


class StudentModel:
    """Interpretable student model wrapper."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize the model.
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.target_type = cfg.target_type
        self.model_type = cfg.model.student.model_type
        
        # Create the underlying model
        self.model_impl = create_model(self.model_type, cfg, self.target_type)
        self.model = self.model_impl.model
    
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
        self.model_impl.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.
        
        Args:
            X: TF-IDF features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        return self.model_impl.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.
        
        Args:
            X: TF-IDF features of shape (n_samples, n_features)
            
        Returns:
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        return self.model_impl.predict_proba(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model.
        
        Returns:
            Dictionary with model statistics
        """
        return self.model_impl.get_model_info()
    
    def get_feature_importances(self) -> np.ndarray:
        """Get feature importances from the model.
        
        Returns:
            Array of feature importances
        """
        return self.model_impl.get_feature_importances()
    
    def get_tree_text(self, feature_names: Optional[List[str]] = None) -> str:
        """Get a text representation of the tree (for tree-based models only).
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            Text representation of the tree
        """
        if self.model_type == "decision_tree":
            return self.model_impl.get_tree_text(feature_names)
        else:
            raise AttributeError(f"get_tree_text is not available for model type: {self.model_type}")
    
    def compute_path_lengths(self, X: np.ndarray) -> np.ndarray:
        """Compute the decision path lengths for samples in X (for tree-based models only).
        
        Args:
            X: TF-IDF features of shape (n_samples, n_features)
            
        Returns:
            Array of path lengths for each sample
        """
        if self.model_type == "decision_tree":
            return self.model_impl.compute_path_lengths(X)
        else:
            raise AttributeError(f"compute_path_lengths is not available for model type: {self.model_type}")
    
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
        
        # Update the model_impl's model reference
        instance.model_impl.model = instance.model
        
        return instance 