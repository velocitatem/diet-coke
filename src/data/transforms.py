from typing import Dict, List, Union, Any
import numpy as np
from transformers import PreTrainedTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize_text(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    truncation: bool = True,
    padding: Union[bool, str] = "max_length"
) -> Dict[str, List[List[int]]]:
    """Tokenize a list of texts using a BERT tokenizer.
    
    Args:
        texts: List of text strings to tokenize
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        truncation: Whether to truncate sequences that are too long
        padding: Padding strategy ('max_length', 'longest', or False)
        
    Returns:
        Dictionary containing 'input_ids' and 'attention_mask'
    """
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=None  # Return Python lists instead of tensors
    )


def create_tfidf_features(
    texts: List[str],
    vectorizer: TfidfVectorizer,
    fit: bool = False
) -> np.ndarray:
    """Create TF-IDF features from a list of texts.
    
    Args:
        texts: List of text strings to vectorize
        vectorizer: Initialized TF-IDF vectorizer
        fit: Whether to fit the vectorizer on these texts
        
    Returns:
        Sparse matrix of TF-IDF features
    """
    if fit:
        return vectorizer.fit_transform(texts)
    else:
        return vectorizer.transform(texts)


def softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature scaling.
    
    Args:
        logits: Raw logits of shape (n_samples, n_classes)
        temperature: Temperature parameter (higher = softer probabilities)
        
    Returns:
        Soft probabilities of shape (n_samples, n_classes)
    """
    # Input validation
    if not isinstance(logits, np.ndarray):
        logits = np.array(logits, dtype=np.float32)
    
    # Handle empty or invalid input
    if logits.size == 0:
        print("Warning: Empty logits array provided to softmax_with_temperature")
        return np.array([])
    
    # Handle NaN/Inf values in input
    if np.isnan(logits).any() or np.isinf(logits).any():
        print("Warning: NaN or Inf values in logits, replacing with finite values")
        logits = np.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Ensure valid temperature
    if temperature <= 0:
        print(f"Warning: Invalid temperature {temperature}, using default of 1.0")
        temperature = 1.0
    
    try:
        # Apply temperature scaling
        logits_scaled = logits / temperature
        
        # Subtract the maximum for numerical stability (prevents overflow)
        logits_shifted = logits_scaled - np.max(logits_scaled, axis=1, keepdims=True)
        
        # Calculate softmax
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Handle any remaining NaN values (should be rare)
        if np.isnan(probs).any():
            print("Warning: NaN values in softmax output, replacing with uniform distribution")
            n_classes = logits.shape[1]
            nan_mask = np.isnan(probs).any(axis=1)
            probs[nan_mask] = 1.0 / n_classes
        
        return probs
    except Exception as e:
        print(f"Error in softmax_with_temperature: {e}")
        # Return uniform distribution as fallback
        return np.ones_like(logits) / logits.shape[1] 