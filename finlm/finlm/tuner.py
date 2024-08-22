import optuna
import wandb
import torch.nn as nn
import torch
from typing import Dict, Any, Tuple, List
from enum import Enum
import logging
from abc import ABC, abstractmethod
import numpy as np
import datasets
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from finlm.finlm.models import PretrainMLM

#TODO: merge the Hyperparam class of downstream, such that we have two Hyperparam tuner classes, each with different
# objectives


class ParameterTypes(Enum):
    HIDDEN_SIZE = "hidden_size"
    OPTIMIZER = "optimizer"
    LEARNING_RATE = "learning_rate"

class AbstractHyperParameter(ABC):
    def __init__(self, name, dtype, low=None, high=None, default=None, options: Tuple[Any]=()):
        """
        Initializes a Hyperparameter instance with the specified attributes.

        Parameters
        ----------
        name : str
            The name of the hyperparameter.
        dtype : type
            The data type of the hyperparameter (e.g., int, float, bool).
        low : Optional[Union[int, float]], optional
            The lower bound for the hyperparameter (default is None).
        high : Optional[Union[int, float]], optional
            The upper bound for the hyperparameter (default is None).
        default : Optional[Union[int, float, bool]], optional
            The default value of the hyperparameter (default is None).
        """

        self.name = name
        self.dtype = dtype
        self.low = low
        self.high = high
        self.default = default

        self.options = options

        if self.low is not None or self.high is not None and len(self.options) > 0:
            raise ValueError("Either take options or a low and a high. Received both")


    @abstractmethod
    def handle_trial(self, trial: optuna.Trial):
        raise NotImplementedError

    @abstractmethod
    def handle_value_assignment(self, value: Any, trainer: PretrainMLM):
        raise NotImplementedError


class LearningRateHyperparam(AbstractHyperParameter):
    def handle_trial(self, trial: optuna.Trial):
        trial.suggest_loguniform(self.name, self.low, self.high)

    def handle_value_assignment(self, value: Any, trainer: PretrainMLM):
        trainer.optimization_config.learning_rate = value
        for g in trainer.optimizer.param_groups:
            g['lr'] = value


class HiddenSizeHyperparam(AbstractHyperParameter):
    def handle_trial(self, trial: optuna.Trial):
        if len(self.options) > 0:
            trial.suggest_categorical(self.name, self.options)
        else:
            trial.suggest_int(self.low, self.high, )


class PretrainingHyperparamTuner:
    def __init__(self, trainer: PretrainMLM, params: Tuple[AbstractHyperParameter],
                 train_steps: int = 30_000, eval_steps: int = 1000):
        self.trainer = trainer
        self.params = params
        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.logger = logging.getLogger(self.__class__.__name__)


   def add_linear_classifier_head_and_freeze(self, children_to_remove: int, last_output_dim: int,
                                             num_classes: int) -> nn.Module:
       linear_head = nn.Linear(last_output_dim, num_classes)
       new_layer_list = list(self.trainer.model.children)[:-children_to_remove]

       for layer in new_layer_list:
           for param in layer.parameters():
               param.requires_grad = False

       new_layer_list.append(linear_head)
       new_model = nn.Sequential(*new_layer_list)

       return new_model

    def train_model(self, steps: int):
        self.logger.info("Running one parameter setting...")
        info_step = 100  # TODO: make variable
        training_metrics = {
            "loss": [],
            "accuracy": [],
            "gradient_norms": [],
            "learning_rates": []
        }

        self.trainer.run_epoch(training_metrics, info_step, steps, verbose=False)

        return training_metrics

    def objective(self, trial: optuna.Trial):
        for hyperparam in self.params:
            hyperparam.handle_trial(trial)
            hyperparam.handle_value_assignment(self.trainer)


        training_metrics = self.train_model(self.train_steps)
        training_summary = self._postprocess_training_metrics(training_metrics)
        eval_metrics = self.evaluate_model(self.eval_steps)



    def _postprocess_training_metrics(self, metrics: Dict[str, List[float]]):
        """
            Post-processes the training metrics to compute summary statistics that provide insights
            into the model's performance and training stability.

            Parameters
            ----------
            metrics : dict
                A dictionary containing lists of metric values collected during training.
                Expected keys are:
                    - "loss": List of loss values recorded at each training step.
                    - "accuracy": List of accuracy values recorded at each training step.

            Returns
            -------
            summary : dict
                A dictionary containing summary statistics of the training metrics. The keys in the
                summary include:
                    - "final_loss": The final loss value at the end of training.
                    - "final_accuracy": The final accuracy value at the end of training.
                    - "mean_loss": The mean of all loss values during training.
                    - "std_loss": The standard deviation of all loss values during training.
                    - "mean_accuracy": The mean of all accuracy values during training.
                    - "std_accuracy": The standard deviation of all accuracy values during training.
                    - "mean_loss_delta": The mean of the changes in loss between consecutive steps.
                    - "std_loss_delta": The standard deviation of the changes in loss between consecutive steps.
                    - "loss_trend_slope" (optional): The slope of the loss trend over time, calculated using
                      a simple linear regression. This is included if there are multiple loss values.

            Notes
            -----
            - The mean and standard deviation of loss deltas (`mean_loss_delta` and `std_loss_delta`)
              provide insight into the stability of the training process.
            - The `loss_trend_slope` can be used to understand the overall direction of the loss
              during training, where a negative slope typically indicates a decreasing loss.
        """

        # Convert lists to numpy arrays for easier mathematical operations
        losses = np.array(metrics["loss"])
        accuracies = np.array(metrics["accuracy"])

        # Calculate the deltas (differences) between consecutive loss values
        loss_deltas = np.diff(losses)

        # Calculate summary statistics for the losses, accuracies, and loss deltas
        summary = {
            "final_loss": losses[-1],  # The last loss in the training run
            "final_accuracy": accuracies[-1],  # The last accuracy in the training run
            "mean_loss": np.mean(losses),  # Average loss over the training run
            "std_loss": np.std(losses),  # Standard deviation of loss
            "mean_accuracy": np.mean(accuracies),  # Average accuracy over the training run
            "std_accuracy": np.std(accuracies),  # Standard deviation of accuracy
            "mean_loss_delta": np.mean(loss_deltas),  # Average change in loss between steps
            "std_loss_delta": np.std(loss_deltas)  # Variability of change in loss between steps
        }

        if len(losses) > 1:
            x = np.arange(len(losses))
            slope, _ = np.polyfit(x, losses, 1)  # Simple linear regression
            summary["loss_trend_slope"] = slope

        return summary

    def evaluate_model(self, steps):
        eval_metrics = {
            "loss": [],
            "accuracy": [],
        }

        self.trainer.evaluate(eval_metrics, num_steps=steps)

        return eval_metrics

    def evaluate_linear_separability(self, dataset: datasets.Dataset, num_classes: int, num_steps: int = 1000):
        """
        Evaluates the linear separability of the model's embeddings by training a linear classifier on top of the frozen model.

        Parameters
        ----------
        dataset : datasets.Dataset
            The dataset to be used for evaluating linear separability.
        num_classes : int
            The number of classes in the classification task.
        num_steps : int, optional
            The number of training steps for the linear classifier, by default 1000.

        Returns
        -------
        float
            The accuracy of the linear classifier, which indicates how linearly separable the embeddings are.
        """
        # Freeze the model and add a linear head
        last_output_dim = self.trainer.model_config.hidden_size  # Assuming hidden_size is the last layer's output dim
        classifier_model = self.add_linear_classifier_head_and_freeze(
            children_to_remove=2,
            last_output_dim=last_output_dim,
            num_classes=num_classes
        )
        classifier_model.to(self.trainer.device)

        # Prepare the dataset and dataloader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Define the optimizer for the classifier
        optimizer = torch.optim.Adam(classifier_model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        classifier_model.train()
        for step in range(num_steps):
            for batch in dataloader:
                inputs, labels = batch["input_ids"].to(self.trainer.device), batch["labels"].to(self.trainer.device)
                outputs = classifier_model(inputs)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    self.logger.info(f"Step {step}, Loss: {loss.item()}")

        # Evaluate the classifier
        classifier_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch["input_ids"].to(self.trainer.device), batch["labels"].to(self.trainer.device)
                outputs = classifier_model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy


def evaluate_svm_separability(self, dataset: datasets.Dataset, num_classes: int, kernel: str = 'linear'):
    """
    Evaluates the separability of the model's embeddings using a Support Vector Machine (SVM).

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to be used for evaluating separability.
    num_classes : int
        The number of classes in the classification task.
    kernel : str, optional
        The kernel type to be used in the SVM (e.g., 'linear', 'rbf'), by default 'linear'.

    Returns
    -------
    float
        The accuracy of the SVM, indicating the separability of the embeddings.
    """
    # Extract embeddings using the frozen model
    embeddings = []
    labels = []
    for batch in DataLoader(dataset, batch_size=32, shuffle=False):
        inputs, batch_labels = batch["input_ids"].to(self.trainer.device), batch["labels"]
        with torch.no_grad():
            model_output = self.trainer.model(inputs)
        embeddings.append(model_output.cpu().numpy())
        labels.append(batch_labels.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Train SVM on the embeddings
    svm = SVC(kernel=kernel)
    svm.fit(embeddings, labels)

    # Predict and evaluate
    predictions = svm.predict(embeddings)
    accuracy = accuracy_score(labels, predictions)

    return accuracy





class DownstreamHyperparamTuner:
    def __init__(self):
        pass