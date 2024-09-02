import torch
import os
import logging
from enum import Enum
from typing import Callable
import wandb
from abc import ABC, abstractmethod


class CallbackTypes(Enum):
    """
    Enum representing the different types of callback events in a training process.

    Attributes:
        BEFORE_TRAINING: Event type triggered before the training process starts.
        AFTER_EVAL: Event type triggered after the evaluation phase.
        AFTER_EPOCH: Event type triggered after the completion of an epoch.
        ON_BATCH_START: Event type triggered at the start of each batch.
        ON_BATCH_END: Event type triggered at the end of each batch.
        ON_EVAL: Event type triggered during the evaluation phase.
    """
    BEFORE_TRAINING = "before_training"
    AFTER_EVAL = "after_eval"
    AFTER_EPOCH = "after_epoch"
    ON_BATCH_START = "on_batch_start"
    ON_BATCH_END = "on_batch_end"
    ON_EVAL = "on_eval"
    AFTER_INIT = "after_init"
    AFTER_LOAD_MODEL = "after_load_model"
    AFTER_OPTIM_LOAD = "after_optim_load"
    BEFORE_BATCH_PROCESSING = "before_batch_processing"
    AFTER_BATCH_PROCESSING = "after_batch_processing"


class AbstractCallback(ABC):
    """
    Abstract base class for defining custom callbacks.

    Attributes:
        type (CallbackTypes): The type of event the callback is associated with.
    """

    def __init__(self, type: CallbackTypes):
        """
        Initializes the callback with a specific event type.

        :param type: The type of the callback event (must be an instance of CallbackTypes).
        """
        self.type = type

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Abstract method that must be implemented by subclasses to define the callback behavior.

        :param args: Positional arguments to be passed to the callback.
        :param kwargs: Keyword arguments to be passed to the callback.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Not implemented yet")

class CallbackManager:
    """
    Manages the registration and execution of callbacks during the training process.

    Attributes:
        callbacks (dict): A dictionary holding lists of callbacks for each event type.
    """

    def __init__(self):
        """
        Initializes the CallbackManager with empty callback lists for each event type.
        """
        self.callbacks = {
            "on_eval": [],
            "before_training": [],
            "after_eval": [],
            "after_epoch": [],
            "on_batch_start": [],
            "on_batch_end": [],
        }

    def add_callback(self, callback: AbstractCallback):
        """
        Registers a callback to a specific event type.

        :param callback: An instance of AbstractCallback to be added.
        :raises ValueError: If the callback's event type is not supported.
        """
        if callback.type.value in self.callbacks:
            self.callbacks[callback.type.value].append(callback)
        else:
            raise ValueError(f"Event {callback.type.value} not supported.")

    def execute_callbacks(self, event, *args, **kwargs):
        """
        Executes all callbacks associated with a specific event.

        :param event: The event type as a string.
        :param args: Positional arguments to be passed to each callback.
        :param kwargs: Keyword arguments to be passed to each callback.
        """
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

    def __init__(self, patience=5, verbose=False, delta=0, mode='min', save_path='checkpoints'):

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
                self.logger.info(
                    f'Validation metric decreased ({self.val_metric_min:.6f} --> {val_metric:.6f}).  Saving model ...')
                self.val_metric_min = val_metric
            elif self.mode == 'max':
                self.logger.info(
                    f'Validation metric increased ({self.val_metric_max:.6f} --> {val_metric:.6f}).  Saving model ...')
                self.val_metric_max = val_metric
        torch.save(model.state_dict(), os.path.join(self.save_path, 'checkpoint.pth'))


class MlMWandBTrackerCallback(AbstractCallback):
    """
    Callback for logging metrics to Weights & Biases (W&B) during the training process.

    Attributes:
        project_name (str): The name of the W&B project.
        api_key (str): The API key for authenticating with W&B.
        params (dict): Additional parameters to be logged as config in W&B.
    """

    def __init__(self, type: CallbackTypes, project_name: str, api_key: str, entity: str = None, name:str = None,
                 **params):
        """
        Initializes the W&B tracker.

        :param type: The type of the callback event (must be an instance of CallbackTypes).
        :param project_name: The name of the W&B project.
        :param api_key: The API key for W&B authentication.
        :param params: Additional parameters to be logged as config in W&B.
        """
        super().__init__(type)

        # Initialize W&B
        wandb.login(key=api_key)
        wandb.init(project=project_name, entity=entity, name=name, reinit=True)

        # Log the initial parameters as config
        for k, v in params.items():
            wandb.config[k] = v

    def log_metrics(self, global_step: int, mlm_loss: float, mlm_accuracy: float, mlm_grad_norm: float,
                    current_lr: float):
        """
        Logs metrics to W&B.

        :param global_step: The current global step of the training process.
        :param mlm_loss: The loss value for the current batch.
        :param mlm_accuracy: The accuracy for the current batch.
        :param mlm_grad_norm: The gradient norm for the current batch.
        :param current_lr: The learning rate for the current batch.
        """
        # Log the metrics to W&B
        wandb.log({
            "step": global_step,
            "mlm_loss": mlm_loss,
            "mlm_accuracy": mlm_accuracy,
            "mlm_grad_norm": mlm_grad_norm,
            "learning_rate": current_lr,
        }, step=global_step)

    def __call__(self, global_step: int, mlm_loss: float, mlm_accuracy: float, mlm_grad_norm: float,
                 current_lr: float):
        """
        Executes the callback, logging the metrics to W&B.

        :param global_step: The current global step of the training process.
        :param mlm_loss: The loss value for the current batch.
        :param mlm_accuracy: The accuracy for the current batch.
        :param mlm_grad_norm: The gradient norm for the current batch.
        :param current_lr: The learning rate for the current batch.
        """
        self.log_metrics(global_step, mlm_loss, mlm_accuracy, mlm_grad_norm, current_lr)