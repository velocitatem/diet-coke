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
        # Ensure all inputs have the same number of samples
        n_samples = len(hard_labels)
        if len(teacher_logits) != n_samples:
            print(f"Warning: Shape mismatch between teacher_logits ({len(teacher_logits)}) and hard_labels ({n_samples})")
            # Truncate to the smaller size
            n_samples = min(n_samples, len(teacher_logits))
            teacher_logits = teacher_logits[:n_samples]
            hard_labels = hard_labels[:n_samples]
            tfidf_features = tfidf_features[:n_samples]
            if texts is not None:
                texts = texts[:n_samples]
        
        # Convert inputs to the correct types
        teacher_logits = np.asarray(teacher_logits, dtype=np.float32)
        hard_labels = np.asarray(hard_labels, dtype=np.int32)
        
        # Check for and handle NaN/Inf values
        if np.isnan(teacher_logits).any() or np.isinf(teacher_logits).any():
            print("Warning: NaN or Inf values detected in teacher logits")
            teacher_logits = np.nan_to_num(teacher_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Apply temperature scaling to get soft probabilities
        soft_probs = softmax_with_temperature(teacher_logits, temperature=self.T)
        
        # Create a DataFrame with metadata
        try:
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
        except Exception as e:
            print(f"Error creating distillation DataFrame: {e}")
            # Create a minimal DataFrame if the above fails
            df = pd.DataFrame({
                'hard_label': hard_labels,
                'sample_weight': 1.0
            })
        
        # Select target for the student model based on configuration
        try:
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
        except Exception as e:
            print(f"Error preparing target values: {e}")
            # Fall back to hard labels if there's an error
            y_target = hard_labels
        
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
        try:
            student_probs = student.predict_proba(tfidf_features)
            student_preds = student.predict(tfidf_features)
            
            # Check for NaN/Inf values in student predictions
            if np.isnan(student_probs).any() or np.isinf(student_probs).any():
                print("Warning: NaN or Inf values detected in student predictions")
                # Replace NaN/Inf values with small/large numbers
                student_probs = np.nan_to_num(student_probs, nan=0.5, posinf=1.0, neginf=0.0)
        except Exception as e:
            print(f"Error during student prediction: {e}")
            # Return default metrics on error
            return {"fidelity/agreement": 0.0, "fidelity/mse": 1.0, "fidelity/cross_entropy": 10.0}
        
        # For classification, teacher_outputs are class probabilities
        if self.target_type == "classification":
            if len(teacher_outputs.shape) == 1:
                # Convert hard labels to one-hot
                teacher_probs = np.zeros((len(teacher_outputs), 2))
                teacher_probs[np.arange(len(teacher_outputs)), teacher_outputs.astype(int)] = 1
                teacher_preds = teacher_outputs.astype(int)
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
        # Use more stable calculation with clipping to prevent numerical issues
        epsilon = 1e-10  # Small constant to avoid log(0)
        student_probs_clipped = np.clip(student_probs, epsilon, 1.0 - epsilon)
        ce = -np.mean(np.sum(teacher_probs * np.log(student_probs_clipped), axis=1))
        
        # Handle any NaN or Inf values in metrics
        if np.isnan(ce) or np.isinf(ce):
            print("Warning: NaN or Inf values in cross entropy, using fallback value")
            ce = 10.0  # Fallback value
        
        if np.isnan(mse) or np.isinf(mse):
            print("Warning: NaN or Inf values in MSE, using fallback value")
            mse = 1.0  # Fallback value
        
        metrics = {
            "fidelity/agreement": float(agreement),
            "fidelity/mse": float(mse),
            "fidelity/cross_entropy": float(ce)
        }
        
        # Log metrics
        if self.writer is not None:
            for key, value in metrics.items():
                # Ensure value is a float and not NaN/Inf before logging
                if isinstance(value, (int, float, np.number)) and not np.isnan(value) and not np.isinf(value):
                    self.writer.add_scalar(key, value, 0)
        
        return metrics 