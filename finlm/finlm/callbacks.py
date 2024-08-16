import torch
import os
import logging
from enum import Enum
from typing import Callable

class CallbackTypes(Enum):
    BEFORE_TRAINING = "before_training"
    AFTER_EVAL = "after_eval"
    AFTER_EPOCH = "after_epoch"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"
    ON_EVAL = "on_eval"


class CallbackManager:
    def __init__(self):
        self.callbacks = {
            "on_eval": [],
            "before_training": [],
            "after_eval": [],
            "after_epoch": [],
            "on_batch_start": [],
            "on_batch_end": [],
        }

    def add_callback(self, event: CallbackTypes, callback: Callable):
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            raise ValueError(f"Event {event} not supported.")

    def execute_callbacks(self, event, *args, **kwargs):
        for callback in self.callbacks.get(event, []):
            callback(*args, **kwargs)

class EarlyStopping:

    """
    Early stopping utility to stop training when a monitored metric has stopped improving.

    This class monitors a validation metric during training and stops training when the metric 
    fails to improve after a given number of epochs (patience). It can operate in both 'min' 
    mode (for metrics that should decrease) and 'max' mode (for metrics that should increase).
    When an improvement is detected, the current model is saved to a specified directory.

    Attributes
    ----------
    patience : int
        Number of epochs to wait after the last improvement in the monitored metric.
    verbose : bool
        If True, logs information about improvements and early stopping.
    delta : float
        Minimum change in the monitored metric to qualify as an improvement.
    mode : str
        One of {'min', 'max'}. Specifies whether to stop training when the metric stops decreasing ('min') or increasing ('max').
    save_path : str
        Directory where model checkpoints are saved.
    counter : int
        Number of epochs without improvement.
    best_score : float or None
        Best score observed for the monitored metric.
    early_stop : bool
        Flag indicating whether early stopping has been triggered.
    logger : logging.Logger
        Logger instance for logging messages related to early stopping.
    val_metric_min : float
        Minimum value of the validation metric observed (used in 'min' mode).
    val_metric_max : float
        Maximum value of the validation metric observed (used in 'max' mode).
    """

    def __init__(self, patience = 5, verbose = False, delta = 0, mode = 'min', save_path = 'checkpoints'):
        
        """
        Initializes the EarlyStopping instance with the specified configuration.

        Parameters
        ----------
        patience : int, optional
            How long to wait after the last time the validation metric improved (default is 5).
        verbose : bool, optional
            If True, prints a message for each validation metric improvement (default is False).
        delta : float, optional
            Minimum change in the monitored quantity to qualify as an improvement (default is 0).
        mode : str, optional
            One of {'min', 'max'}. Specifies whether to stop training when the metric stops decreasing ('min') or increasing ('max') (default is 'min').
        save_path : str, optional
            Directory where model checkpoints will be saved (default is 'checkpoints').
        """

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path
        self.logger = logging.getLogger(self.__class__.__name__)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if mode == 'min':
            self.val_metric_min = float('inf')
        elif mode == 'max':
            self.val_metric_max = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, val_metric, model):

        """
        Checks the validation metric and determines if early stopping should be triggered.

        This method is called after each epoch to monitor the validation metric. It compares 
        the current validation metric to the best score observed so far, and if an improvement 
        is detected, the model is saved. If no improvement is detected for a number of epochs 
        equal to `patience`, early stopping is triggered.

        Parameters
        ----------
        val_metric : float
            The current value of the validation metric to be monitored.
        model : torch.nn.Module
            The model to be saved if the validation metric improves.
        """
        
        if self.mode == 'min':
            score = -val_metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_metric, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_metric, model)
                self.counter = 0
        elif self.mode == 'max':
            score = val_metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_metric, model)
            elif score < self.best_score - self.delta:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_metric, model)
                self.counter = 0
        


    def save_checkpoint(self, val_metric, model):
        
        """
        Saves the model when the validation metric improves.

        This method saves the current state of the model if the validation metric shows improvement.
        It also logs the improvement if `verbose` is set to True.

        Parameters
        ----------
        val_metric : float
            The current value of the validation metric that triggered the checkpoint save.
        model : torch.nn.Module
            The model to be saved.
        """
        
        if self.verbose:
            if self.mode == 'min':
                self.logger.info(f'Validation metric decreased ({self.val_metric_min:.6f} --> {val_metric:.6f}).  Saving model ...')
                self.val_metric_min = val_metric
            elif self.mode == 'max':
                self.logger.info(f'Validation metric increased ({self.val_metric_max:.6f} --> {val_metric:.6f}).  Saving model ...')
                self.val_metric_max = val_metric
        torch.save(model.state_dict(), os.path.join(self.save_path, 'checkpoint.pth'))