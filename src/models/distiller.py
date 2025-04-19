from typing import Dict, Any, Tuple, List, Optional, Union
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_sample_weight
from omegaconf import DictConfig
import lightning as L
from torch.utils.tensorboard import SummaryWriter

from src.data.transforms import softmax_with_temperature
from src.models.teacher import TeacherModel
from src.models.student import StudentModel


class Distiller:
    """Knowledge distillation from BERT to Decision Tree."""
    
    def __init__(self, cfg: DictConfig, writer: Optional[SummaryWriter] = None):
        """Initialize the distiller.
        
        Args:
            cfg: Configuration object
            writer: Optional TensorBoard writer
        """
        self.cfg = cfg
        # Access distillation parameters directly from the config
        self.T = cfg.T
        self.alpha = cfg.alpha
        self.use_proba = cfg.use_proba
        self.balanced_weights = cfg.balanced_weights
        self.target_type = cfg.target_type
        self.writer = writer
    
    def extract_teacher_outputs(
        self, 
        teacher: TeacherModel, 
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract logits and probabilities from the teacher model.
        
        Args:
            teacher: Fine-tuned teacher model
            dataloader: DataLoader with data
            
        Returns:
            Tuple of (logits, labels) as numpy arrays
        """
        # Make sure the teacher model is in eval mode
        teacher.eval()
        
        # Get the device of the teacher model
        device = next(teacher.parameters()).device
        print(f"Teacher model is on device: {device}")
        
        logits, labels = teacher.get_logits(dataloader)
        return logits.numpy(), labels.numpy()
    
    def prepare_distillation_data(
        self, 
        tfidf_features: np.ndarray,
        teacher_logits: np.ndarray,
        hard_labels: np.ndarray,
        texts: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare data for distillation.
        
        Args:
            tfidf_features: TF-IDF features of shape (n_samples, n_features)
            teacher_logits: Logits from teacher of shape (n_samples, n_classes)
            hard_labels: Ground truth labels of shape (n_samples,)
            texts: Optional list of original texts
            
        Returns:
            Tuple of (DataFrame with metadata, TF-IDF features, target values)
        """
        # Apply temperature scaling to get soft probabilities
        soft_probs = softmax_with_temperature(teacher_logits, temperature=self.T)
        
        # Create a DataFrame with metadata
        df = pd.DataFrame({
            'hard_label': hard_labels,
            'soft_prob_0': soft_probs[:, 0],
            'soft_prob_1': soft_probs[:, 1],
        })
        
        if texts is not None:
            df['text'] = texts
        
        # Compute sample weights if needed
        if self.balanced_weights:
            sample_weights = compute_sample_weight('balanced', hard_labels)
            df['sample_weight'] = sample_weights
        else:
            df['sample_weight'] = 1.0
        
        # Select target for the student model based on configuration
        if self.target_type == "classification":
            if self.use_proba:
                # Classification with soft targets
                y_target = soft_probs if self.alpha == 1.0 else hard_labels
            else:
                # Classification with hard targets
                y_target = hard_labels
        else:
            # Regression: use only the probability of the positive class
            y_target = soft_probs[:, 1]
        
        return df, tfidf_features, y_target
    
    def fit_student(
        self,
        tfidf_features: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> StudentModel:
        """Fit the student model.
        
        Args:
            tfidf_features: TF-IDF features of shape (n_samples, n_features)
            targets: Target values for the student model
            sample_weights: Optional sample weights
            
        Returns:
            Trained student model
        """
        # Initialize student model
        student = StudentModel(self.cfg)
        
        # Fit the model
        student.fit(tfidf_features, targets, sample_weight=sample_weights)
        
        # Log tree information
        tree_info = student.get_model_info()
        if self.writer is not None:
            for key, value in tree_info.items():
                # Only log numeric values to TensorBoard
                if isinstance(value, (int, float, bool, np.number)):
                    self.writer.add_scalar(f"tree/{key}", float(value), 0)
        
        return student
    
    def distill(
        self,
        teacher: TeacherModel,
        dataloader: DataLoader,
        tfidf_features: np.ndarray,
        texts: Optional[List[str]] = None
    ) -> Tuple[StudentModel, pd.DataFrame]:
        """Perform distillation from teacher to student.
        
        Args:
            teacher: Fine-tuned teacher model
            dataloader: DataLoader with training data
            tfidf_features: TF-IDF features of shape (n_samples, n_features)
            texts: Optional list of original texts
            
        Returns:
            Tuple of (trained student model, distillation DataFrame)
        """
        # Extract teacher outputs
        teacher_logits, hard_labels = self.extract_teacher_outputs(teacher, dataloader)
        
        # Prepare distillation data
        distill_df, X, y = self.prepare_distillation_data(
            tfidf_features, teacher_logits, hard_labels, texts
        )
        
        # Get sample weights if needed
        sample_weights = distill_df['sample_weight'].values if self.balanced_weights else None
        
        # Fit student model
        student = self.fit_student(X, y, sample_weights)
        
        return student, distill_df
    
    def evaluate_fidelity(
        self, 
        student: StudentModel, 
        teacher_outputs: np.ndarray, 
        tfidf_features: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate the fidelity of the student to the teacher.
        
        Args:
            student: Trained student model
            teacher_outputs: Outputs from the teacher model
            tfidf_features: TF-IDF features
            
        Returns:
            Dictionary with fidelity metrics
        """
        # Get student predictions
        student_probs = student.predict_proba(tfidf_features)
        student_preds = student.predict(tfidf_features)
        
        # For classification, teacher_outputs are class probabilities
        if self.target_type == "classification":
            if len(teacher_outputs.shape) == 1:
                # Convert hard labels to one-hot
                teacher_probs = np.zeros((len(teacher_outputs), 2))
                teacher_probs[np.arange(len(teacher_outputs)), teacher_outputs] = 1
                teacher_preds = teacher_outputs
            else:
                # Use soft probabilities
                teacher_probs = teacher_outputs
                teacher_preds = np.argmax(teacher_probs, axis=1)
        else:
            # For regression, convert to probabilities and class labels
            if len(teacher_outputs.shape) == 1:
                teacher_probs = np.vstack((1 - teacher_outputs, teacher_outputs)).T
                teacher_preds = (teacher_outputs > 0.5).astype(int)
            else:
                teacher_probs = teacher_outputs
                teacher_preds = np.argmax(teacher_probs, axis=1)
        
        # Calculate fidelity metrics
        agreement = np.mean(student_preds == teacher_preds)
        
        # Mean squared error between soft probabilities
        mse = np.mean((student_probs - teacher_probs) ** 2)
        
        # Cross-entropy between teacher and student distributions
        epsilon = 1e-15  # Small constant to avoid log(0)
        ce = -np.mean(np.sum(teacher_probs * np.log(student_probs + epsilon), axis=1))
        
        metrics = {
            "fidelity/agreement": agreement,
            "fidelity/mse": mse,
            "fidelity/cross_entropy": ce
        }
        
        # Log metrics
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, 0)
        
        return metrics 