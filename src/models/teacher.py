from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torchmetrics

class TeacherModel(pl.LightningModule):
    """BERT teacher model for sequence classification."""
    
    def __init__(self, cfg: Dict[str, Any]):
        """Initialize the model.
        
        Args:
            cfg: Configuration dictionary with model parameters
        """
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        # Load BERT configuration
        self.config = AutoConfig.from_pretrained(
            cfg.model.teacher.name,
            num_labels=cfg.model.teacher.num_labels,
            cache_dir=None
        )
        
        # Load pre-trained BERT model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.teacher.name, 
            config=self.config,
            cache_dir=None
        )
        
        # Set up metrics for logging
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            labels: Optional labels of shape (batch_size,)
            
        Returns:
            Dictionary with model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def training_step(self, batch: List[torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Training step.
        
        Args:
            batch: Batch of data containing (input_ids, attention_mask, labels)
            batch_idx: Index of the batch
            
        Returns:
            Dictionary with loss and log information
        """
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        acc = self.train_acc(preds, labels)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        
        return {'loss': loss, 'preds': preds, 'labels': labels}
    
    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step.
        
        Args:
            batch: Batch of data containing (input_ids, attention_mask, labels)
            batch_idx: Index of the batch
            
        Returns:
            Dictionary with validation metrics
        """
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        acc = self.val_acc(preds, labels)
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        
        return {'val_loss': loss, 'val_preds': preds, 'val_labels': labels, 'val_logits': logits}
    
    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step.
        
        Args:
            batch: Batch of data containing (input_ids, attention_mask, labels)
            batch_idx: Index of the batch
            
        Returns:
            Dictionary with test metrics
        """
        input_ids, attention_mask, labels = batch
        outputs = self(input_ids, attention_mask, labels)
        
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        # Log metrics
        acc = self.test_acc(preds, labels)
        self.log('test/loss', loss)
        self.log('test/acc', acc)
        
        return {'test_loss': loss, 'test_preds': preds, 'test_labels': labels, 'test_logits': logits}
    
    def predict_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step.
        
        Args:
            batch: Batch of data containing (input_ids, attention_mask, [labels])
            batch_idx: Index of the batch
            
        Returns:
            Dictionary with predictions
        """
        input_ids, attention_mask = batch[:2]  # Labels might be present or not
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        return {'preds': preds, 'logits': logits}
    
    def configure_optimizers(self) -> Tuple[List[AdamW], List[Dict[str, Any]]]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Tuple of (optimizers, schedulers)
        """
        # Extract parameters that need different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.model.teacher.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Create AdamW optimizer
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.cfg.model.teacher.lr,
            eps=self.cfg.model.teacher.adam_epsilon
        )
        
        # Create scheduler
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.cfg.model.teacher.warmup_steps
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val/loss",
        }
        
        return [optimizer], [scheduler_config]
    
    def get_logits(self, dataloader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get logits and labels from a dataloader.
        
        Args:
            dataloader: DataLoader with data
            
        Returns:
            Tuple of (logits, labels)
        """
        self.model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels.cpu())
        
        return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0) 