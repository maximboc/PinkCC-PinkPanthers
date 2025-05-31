import numpy as np
import torch

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights:
                self.load_checkpoint(model)
                
    def save_checkpoint(self, model):
        """Save model when validation loss decreases."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
            
    def load_checkpoint(self, model):
        """Load best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
