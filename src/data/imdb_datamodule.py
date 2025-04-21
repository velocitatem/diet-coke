from typing import Dict, Optional, Tuple, Any, List, Union
import os
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl
from datasets import load_dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from src.data.transforms import tokenize_text, create_tfidf_features


class IMDBDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for the IMDB dataset."""

    def __init__(self, cfg: DictConfig):
        """Initialize the data module.
        
        Args:
            cfg: Configuration object containing dataset parameters
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg.dataset.name
        self.n_train = cfg.dataset.n_train
        self.n_test = cfg.dataset.n_test
        self.seed = cfg.dataset.seed
        self.batch_size = cfg.train.batch_size
        self.eval_batch_size = cfg.eval.batch_size
        
        # Tokenizer for BERT
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.tokenizer.name,
                use_fast=True,
                cache_dir=os.environ.get("MODEL_CACHE_DIR", None)
            )
        except OSError as e:
            # If tokenizer files are not found, try to download them
            print(f"Tokenizer not found locally. Downloading {cfg.tokenizer.name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.tokenizer.name,
                use_fast=True,
                cache_dir=os.environ.get("MODEL_CACHE_DIR", None),
                local_files_only=False  # Allow downloading if not found locally
            )
        
        # TF-IDF vectorizer for Decision Tree
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.vectorizer.max_features,
            ngram_range=tuple(cfg.vectorizer.ngram_range),
            stop_words=cfg.vectorizer.stop_words,
            lowercase=cfg.vectorizer.lowercase
        )
        
        # Data storage
        self.imdb_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.tfidf_features = None

    def prepare_data(self) -> None:
        """Download the IMDB dataset and tokenizer."""
        # Download the dataset
        load_dataset(
            self.dataset_name,
            cache_dir=os.environ.get("DATA_CACHE_DIR", None)
        )
        
        # Download the tokenizer
        try:
            AutoTokenizer.from_pretrained(
                self.cfg.tokenizer.name,
                use_fast=True,
                cache_dir=os.environ.get("MODEL_CACHE_DIR", None),
                local_files_only=False
            )
        except Exception as e:
            print(f"Error downloading tokenizer: {e}")
            raise

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the datasets for training, validation, and testing.
        
        Args:
            stage: Stage to set up ('fit', 'validate', 'test', or 'predict')
        """
        # Load datasets
        self.imdb_dataset = load_dataset(
            self.dataset_name,
            cache_dir=os.environ.get("DATA_CACHE_DIR", None)
        )
        
        # For BERT fine-tuning: Create train/val split from the train set
        if stage == 'fit' or stage is None:
            # Sample n_train examples from the train set
            train_val_data = self.imdb_dataset['train'].shuffle(seed=self.seed)
            train_val_data = train_val_data.select(range(min(self.n_train + 1000, len(train_val_data))))
            
            # Create train/val split stratified by label
            train_val_split = train_val_data.train_test_split(
                test_size=0.1,  # 10% for validation
                seed=self.seed,
                stratify_by_column='label'
            )
            
            # Convert to DataFrames for easier handling
            self.train_df = pd.DataFrame(train_val_split['train'])
            self.val_df = pd.DataFrame(train_val_split['test'])
            
            # Create tokenized datasets for BERT
            self.train_dataset = self._create_bert_dataset(self.train_df)
            self.val_dataset = self._create_bert_dataset(self.val_df)
            
            # Fit TF-IDF vectorizer on training texts
            self.tfidf_features = create_tfidf_features(
                self.train_df['text'].tolist(),
                self.vectorizer,
                fit=True
            )
            
            # Save the fitted vectorizer
            self.save_vectorizer(self.cfg.paths.vectorizer)
        
        # For evaluation: Use the test set
        if stage == 'test' or stage is None:
            # Sample n_test examples from the test set
            test_data = self.imdb_dataset['test'].shuffle(seed=self.seed)
            test_data = test_data.select(range(min(self.n_test, len(test_data))))
            
            self.test_df = pd.DataFrame(test_data)
            self.test_dataset = self._create_bert_dataset(self.test_df)
            
            # Load the fitted vectorizer if it exists
            if os.path.exists(self.cfg.paths.vectorizer):
                self.load_vectorizer(self.cfg.paths.vectorizer)
            else:
                raise ValueError(f"Vectorizer not found at {self.cfg.paths.vectorizer}. Please run training first.")
    
    def _create_bert_dataset(self, df: pd.DataFrame) -> TensorDataset:
        """Create a TensorDataset from a DataFrame for BERT.
        
        Args:
            df: DataFrame containing 'text' and 'label' columns
            
        Returns:
            TensorDataset with input_ids, attention_mask, and labels
        """
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Tokenize texts
        tokenized = tokenize_text(
            texts,
            self.tokenizer,
            max_length=self.cfg.tokenizer.max_length,
            truncation=self.cfg.tokenizer.truncation,
            padding=self.cfg.tokenizer.padding
        )
        
        # Create tensors
        input_ids = torch.tensor(tokenized['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(tokenized['attention_mask'], dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return TensorDataset(input_ids, attention_mask, labels)
    
    def get_tfidf_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Get the TF-IDF features and labels for the train set.
        
        Returns:
            Tuple containing the train DataFrame and TF-IDF features
        """
        return self.train_df, self.tfidf_features
    
    def create_test_tfidf_features(self) -> np.ndarray:
        """Create TF-IDF features for the test set.
        
        Returns:
            TF-IDF features for the test set
        """
        return create_tfidf_features(
            self.test_df['text'].tolist(),
            self.vectorizer,
            fit=False
        )
    
    def train_dataloader(self) -> DataLoader:
        """Create the training data loader.
        
        Returns:
            DataLoader for the training data
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create the validation data loader.
        
        Returns:
            DataLoader for the validation data
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create the test data loader.
        
        Returns:
            DataLoader for the test data
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def save_tokenizer(self, save_dir: str) -> None:
        """Save the tokenizer to disk.
        
        Args:
            save_dir: Directory to save the tokenizer to
        """
        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
    
    def save_vectorizer(self, save_path: str) -> None:
        """Save the TF-IDF vectorizer to disk.
        
        Args:
            save_path: Path to save the vectorizer to
        """
        import pickle
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load_vectorizer(self, vectorizer_path: str) -> None:
        """Load a fitted TF-IDF vectorizer from disk.
        
        Args:
            vectorizer_path: Path to the saved vectorizer
        """
        import pickle
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f) 