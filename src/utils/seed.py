import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic operations in PyTorch
    # Note: This may slow down training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        # For PyTorch 1.8+
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        # For older PyTorch versions
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"  # For CUDA 10.2+
        
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed) 