#!/usr/bin/env python
"""
Script for fine-tuning BERT on IMDB dataset.
"""
import os
import sys
import json
from typing import Dict, Any
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig, OmegaConf

# Add the project root to the Python path if not already added
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.imdb_datamodule import IMDBDataModule
from src.models.teacher import TeacherModel
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function for training the teacher model.
    
    Args:
        cfg: Configuration object
    """
    # Set random seed for reproducibility
    set_seed(cfg.train.seed)
    
    # Set up logging
    logger, console, tb_writer = setup_logging(cfg)
    logger.info("Starting fine-tuning of BERT teacher model...")
    
    # Create output directories
    os.makedirs(cfg.paths.checkpoints, exist_ok=True)
    
    # Print config
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Initialize data module
    logger.info("Initializing data module...")
    datamodule = IMDBDataModule(cfg)
    
    # Initialize model
    logger.info("Initializing teacher model...")
    model = TeacherModel(cfg)
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoints,
            filename="teacher",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=cfg.train.early_stopping_patience,
            mode="min",
        ),
    ]
    
    # Set up Lightning trainer
    device = "auto" if cfg.train.device == "auto" else cfg.train.device
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=device,
        logger=pl.loggers.TensorBoardLogger(save_dir=cfg.logging.tensorboard_log_dir),
        callbacks=callbacks,
        precision=16 if cfg.train.fp16 else 32,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        deterministic=True,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    
    # Save tokenizer
    logger.info("Saving tokenizer...")
    datamodule.save_tokenizer(os.path.join(cfg.output_dir, "artifacts/tokenizer"))
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.test(model, datamodule=datamodule)[0]
    
    # Log and save results
    metrics_file = os.path.join(cfg.output_dir, "val_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info(f"Test metrics: {test_results}")
    logger.info(f"Model saved to {cfg.paths.teacher_model}")
    logger.info(f"Tokenizer saved to {cfg.output_dir}/artifacts/tokenizer")
    logger.info(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main() 