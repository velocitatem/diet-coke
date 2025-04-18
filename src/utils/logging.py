import os
import sys
import logging
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf


def setup_logging(cfg: DictConfig) -> tuple[logging.Logger, Console, Optional[SummaryWriter]]:
    """Set up logging with Rich console and TensorBoard.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (logger, console, tensorboard_writer)
    """
    # Create log directory
    log_dir = cfg.logging.tensorboard_log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up Rich console for pretty terminal output
    console = Console()
    
    # Configure logging with Rich handler
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    
    logger = logging.getLogger("nlp-distil-bert-tree")
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Log the configuration
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    
    # Also add config as text to TensorBoard
    writer.add_text("config", f"```yaml\n{OmegaConf.to_yaml(cfg)}\n```")
    
    return logger, console, writer


def log_metrics(writer: SummaryWriter, metrics: dict, step: int, prefix: str = "") -> None:
    """Log metrics to TensorBoard.
    
    Args:
        writer: TensorBoard writer
        metrics: Dictionary of metrics to log
        step: Step or epoch number
        prefix: Optional prefix for metric names
    """
    for name, value in metrics.items():
        if prefix:
            name = f"{prefix}/{name}"
        writer.add_scalar(name, value, step) 