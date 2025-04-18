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
    """Apply temperature scaling and softmax to logits.
    
    Args:
        logits: Array of shape (n_samples, n_classes)
        temperature: Temperature parameter (T > 1 makes distribution softer)
        
    Returns:
        Array of softmax probabilities with the same shape as logits
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # For numerical stability, subtract the max from each row
    exps = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    
    # Normalize
    return exps / np.sum(exps, axis=1, keepdims=True) 