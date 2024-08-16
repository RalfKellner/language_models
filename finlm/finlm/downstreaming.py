import optuna
from datasets import Dataset
from finlm.config import FintuningConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from finlm.dataset import FinetuningDataset, FinetuningDocumentDataset, collate_fn_fixed_sequences
from finlm.callbacks import EarlyStopping
from typing import Dict, Any, Union
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Hyperparameter:

    """
    A class representing a hyperparameter with a specified type and optional range.

    This class is used to define hyperparameters that can be used in various machine 
    learning or deep learning models. Each hyperparameter has a name, data type, and 
    optional range (low, high) along with a default value.

    Attributes
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

    Methods
    -------
    from_dict(data: Dict[str, Any]) -> 'Hyperparameter'
        Creates an instance of the Hyperparameter class from a dictionary.
    """

    def __init__(self, name, dtype, low = None, high = None, default = None):

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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hyperparameter':

        """
        Creates an instance of the Hyperparameter class from a dictionary.

        This method converts the data type string to the corresponding Python type 
        (e.g., "int" to int) and initializes the Hyperparameter instance with the 
        provided values.

        Parameters
        ----------
        data : Dict[str, Any]
            A dictionary containing the configuration parameters.

        Returns
        -------
        Hyperparameter
            An instance of Hyperparameter initialized with the provided data.

        Raises
        ------
        ValueError
            If the dtype in the dictionary is not supported.
        """

        dtype_mapper = {
            "int": int,
            "float": float,
            "bool": bool
        }

        dtype = dtype_mapper.get(data["dtype"])

        return cls(name = data["name"], dtype = dtype, low = data["low"], high = data["high"], default = data["default"])
    

class FinetuningEncoderClassifier:

    """
    A class for fine-tuning an encoder-based classifier model, specifically for sequence classification tasks using the ELECTRA architecture.

    This class supports tasks such as regression, binary classification, and multi-class classification.
    It provides functionality for training the model with cross-validation, hyperparameter optimization using Optuna, 
    and final evaluation of the trained model.

    The methods are written bottom to top which means the first method here, is the one which uses the other methods below to find the 
    best set of hyperparameters for a cross validated training data set. Once the optimization is finished the best model and its 
    performance metrics as well as the configuation of the best model search are saved.

    Attributes
    ----------
    config : FintuningConfig
        The configuration object containing hyperparameters and paths required for fine-tuning.
    device : torch.device
        The device (CPU or GPU) on which the model will be trained and evaluated.
    dataset : FinetuningDataset
        The dataset object used for fine-tuning, created from the input dataset and config.
    model_path : str
        Path to the pre-trained model that will be fine-tuned.
    num_labels : int
        Number of labels for the classification task.
    task : str
        The type of task being performed: "regression", "binary_classification", or "multi_classification".
    n_epochs : Hyperparameter
        The number of training epochs, represented as a Hyperparameter object.
    learning_rate : Hyperparameter
        The learning rate for the optimizer, represented as a Hyperparameter object.
    classifier_dropout : Hyperparameter
        The dropout rate used in the classifier layer, represented as a Hyperparameter object.
    warmup_step_fraction : Hyperparameter
        The fraction of warmup steps during training, represented as a Hyperparameter object.
    use_gradient_clipping : Hyperparameter
        A boolean indicating whether to use gradient clipping, represented as a Hyperparameter object.
    save_path : str
        The directory where the trained model and other outputs will be saved.
    logger : logging.Logger
        Logger for recording information during training and evaluation.
    model_loader: callable
        A callable which receives a model path of model name from huggingface, the number of labels and a classifier dropout rate

    Methods
    -------
    train_optuna_optimized_cv_model(n_trials: int)
        Trains the model using cross-validation with hyperparameters optimized by Optuna.
    optuna_optimize(n_trials: int = 10) -> Tuple[Dict[str, Any], float]
        Optimizes hyperparameters using Optuna and returns the best parameters and score.
    optuna_objective(trial) -> float
        Defines the objective function for Optuna hyperparameter optimization.
    cross_validate(
        n_folds: int, 
        training_data, 
        training_batch_size: int, 
        validation_batch_size: int, 
        n_epochs: int, 
        learning_rate: float, 
        classifier_dropout: float, 
        warmup_step_fraction: float, 
        use_gradient_clipping: bool
    ) -> float
        Performs cross-validation on the training data and returns the average score across folds.
    train(
        training_data, 
        validation_data, 
        n_epochs: int, 
        learning_rate: float, 
        classifier_dropout: float, 
        warmup_step_fraction: float, 
        use_gradient_clipping: bool, 
        save_best_model_path: str
    ) -> float
        Trains the model on the training data and validates it on the validation data, with early stopping and checkpointing.
    final_evaluation(finetuned_model_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]
        Evaluates the final trained model on both the training and test datasets, returning performance metrics.
    _determine_scores(true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, Any]
        Computes and logs performance metrics based on the true and predicted labels.
    _load_model(classifier_dropout: float, save_path: str = None) -> Any
        Loads an encoder classification model with the specified dropout rate, optionally from a saved checkpoint as defined by the model_loader callable.
    """

    def __init__(
            self,
            config: FintuningConfig,
            device: torch.device,
            dataset: Union[FinetuningDataset, FinetuningDocumentDataset],
            model_loader: callable
        ):

        """
        Initializes the FinetuningEncoderClassifier with the provided configuration, device, and dataset.

        Parameters
        ----------
        config : FintuningConfig
            The configuration object containing hyperparameters and paths required for fine-tuning.
        device : torch.device
            The device (CPU or GPU) on which the model will be trained and evaluated.
        dataset : Dataset
            The dataset to be used for fine-tuning.
        
        Raises
        ------
        ValueError
            If the save path already exists, indicating the model has already been trained.
        """
    
        self.config = config
        self.model_path = self.config.model_path
        self.num_labels = self.config.num_labels
        if self.num_labels == 1:
            # check if the argmax logic from the current implementation can be used for regression type problems as well
            self.task = "regression"
        elif self.num_labels == 2:
            self.task = "binary_classification"
        else:
            self.task = "multi_classification"
        self.device = device
        # this is the special case with the sequence embedding head
        if isinstance(dataset, FinetuningDocumentDataset):
            self.aggregated_document_model = True
            self.dataset = dataset
            self.collate_fn = lambda x: collate_fn_fixed_sequences(x, max_sequences = self.config.max_sequences)
        else:
        # common case
            self.aggregated_document_model = False
            self.dataset = dataset
            self.collate_fn = None

        self.train_size = int(len(self.dataset) * self.config.training_data_fraction)
        self.training_data = self.dataset.select(range(self.train_size))
        self.test_data = self.dataset.select(range(self.train_size, len(self.dataset)))

        self.batch_size = self.config.batch_size
        self.n_splits = self.config.n_splits
        self.early_stopping_patience = self.config.early_stopping_patience
        # each hyperparameter has a name, dtype, low, high and default value, the default value is used for the first trial
        # depending on the data type, trials for all further trials are sampled according to the appropriate data type
        self.n_epochs = Hyperparameter.from_dict(self.config.n_epochs)
        self.learning_rate = Hyperparameter.from_dict(self.config.learning_rate)
        self.classifier_dropout = Hyperparameter.from_dict(self.config.classifier_dropout)
        self.warmup_step_fraction = Hyperparameter.from_dict(self.config.warmup_step_fraction)
        self.use_gradient_clipping = Hyperparameter.from_dict(self.config.use_gradient_clipping)
        self.save_path = self.config.save_path
        if os.path.exists(self.save_path):
            raise ValueError("It seems you already trained this model, check the save path or delete the current one.")
        self.logger = logging.getLogger(self.__class__.__name__)
        # this is logged to examine if training and test label balances are approximately equal
        # for regression, we need to take a look at descriptive statistics instead of class occurrences
        self.logger.info(f"Counting occurences of labels...")
        self.logger.info(f"Occurence of labels for training data: {self.training_data.num_labels()}")
        self.logger.info(f"Occurence of labels for test data: {self.test_data.num_labels()}")

        # this is a callable which loads the model from a pretrained model before training the hyperparameters
        # during the training it is used to initialize new classifier models with a specified classifier dropout
        # it also includes the possibility to load parameters from a checkpoint.pth file. This is included and mostly
        # used during training to load the best model over all epochs, especially, if the early stopping mechanism is met
        # the model_loader is defined in the finetuning specific python scripts
        self.model_loader = model_loader


    # this function combines most of the methods from below, it uses a optuna hyperparameter search for an optuna objective
    # which derives a performance from a cross validation over training data where each split is trained using the train method
    # so from top to bottom we have: train_optuna_optimized_cv_model -> optuna_optimize -> optuna_objective -> cross_validate -> train
    # so maybe it helps to start from bottom to the top, if you want to understand the whole training mechanism
    # after hyperparameters have been optimized, the model is trained on the full training sample and evaluated on the test sample
    # the final model and its performance is saved in the self.save_path which is set in the yaml-configuration file
    def train_optuna_optimized_cv_model(self, n_trials):

        """
        Trains the model using cross-validation with hyperparameters optimized by Optuna.

        Parameters
        ----------
        n_trials : int
            The number of trials for Optuna optimization.

        This method:
        - Optimizes hyperparameters using Optuna.
        - Trains the model on the full training dataset.
        - Saves the final trained model and evaluation metrics.
        """

        # derive the best hyperparameters for a cross validation metric, only training data is used for cross validation
        best_params = self.optuna_optimize(n_trials = n_trials)

        with open(os.path.join(self.save_path, "best_hyperparameters.json"), "w") as file:
            json.dump(best_params, file, indent = 4)

        # once the hyperparameters are found the model is trained on the full training data set and evaluated for the new and unseen test data
        full_training_split = DataLoader(self.training_data, batch_size = self.batch_size, shuffle = False, collate_fn = self.collate_fn)
        test_split = DataLoader(self.test_data, batch_size = self.batch_size, shuffle = False, collate_fn = self.collate_fn)

        self.train(
            full_training_split,
            test_split,
            n_epochs = best_params["n_epochs"],
            learning_rate = best_params["learning_rate"],
            classifier_dropout = best_params["classifier_dropout"],
            warmup_step_fraction = best_params["warmup_step_fraction"],
            use_gradient_clipping = best_params["use_gradient_clipping"],
            save_best_model_path = self.save_path           
        )

        # self._load_model initializes the model from pretraining
        model = self._load_model(classifier_dropout = best_params["classifier_dropout"])        
        self.logger.info(f"Loading finetuned model from checkpoint.")
        # during the training call from above parameters have been saved in the ..../self.save_path/checkpoint.pth file, these are loaded here
        model.load_state_dict(torch.load(os.path.join(self.save_path, "checkpoint.pth")))
        # this saves the model in the common huggingface format, creating a folder in ..../self.save_path/fintuned_model which includes the model config and parameter as tensors
        model.save_pretrained(os.path.join(self.save_path, "finetuned_model"))
        # this saves the configuation info from the hyperparameter tuning
        self.config.to_json(os.path.join(self.save_path, "finetuning_config.json"))
        # save final scores
        training_scores, test_scores = self.final_evaluation(os.path.join(self.save_path, "finetuned_model"), classifier_dropout=best_params["classifier_dropout"])
        
        if self.task == "multi_classification":
            training_scores["precision_scores"] = training_scores["precision_scores"].tolist()
            training_scores["recall_scores"] = training_scores["recall_scores"].tolist()
            test_scores["precision_scores"] = test_scores["precision_scores"].tolist()
            test_scores["recall_scores"] = test_scores["recall_scores"].tolist()
        
        with open(os.path.join(self.save_path, "training_scores.json"), "w") as file:
            json.dump(training_scores, file, indent = 4)

        with open(os.path.join(self.save_path, "test_scores.json"), "w") as file:
            json.dump(test_scores, file, indent = 4)


    def optuna_optimize(self, n_trials = 10):
            
        """
        Optimizes hyperparameters using Optuna and returns the best parameters and score.

        Parameters
        ----------
        n_trials : int, optional
            The number of trials for Optuna optimization (default is 10).

        Returns
        -------
        Tuple[Dict[str, Any], float]
            The best hyperparameters and the corresponding score.
        """

        # creates an optuna hyperparameter optimization using the training data, currently hyperparameter optimization is supposed to
        # maximize the f1_score for classification and the r2_score for regression tasks
        # the first trial uses default values which can be set in the config file
            
        study = optuna.create_study(direction = "maximize")
        study.optimize(self.optuna_objective, n_trials = n_trials)  

        # if the best trial was the first with default parameters, we save them    
        if study.best_trial.number == 0:
            hyperparameters = [self.n_epochs, self.learning_rate, self.classifier_dropout, self.warmup_step_fraction, self.use_gradient_clipping]
            best_params = {}
            for hyperparameter in hyperparameters:
                best_params[hyperparameter.name] = hyperparameter.default
        # otherwise we save the ones from the best trial after the first trial
        else:
            best_params = study.best_params
        best_params["best_value"] = study.best_value 
        
        return best_params


    def optuna_objective(self, trial):

        """
        Defines the objective function for Optuna hyperparameter optimization.

        Parameters
        ----------
        trial : optuna.Trial
            A single trial object for Optuna optimization.

        Returns
        -------
        float
            The cross-validation score for the current trial's hyperparameters.
        
        Raises
        ------
        ValueError
            If the data type of a hyperparameter is not one of float, int, or bool.
        """

        # these are the current hyperparameters, we may want to change this towards a higher level of generalizaton and flexibility
        # by automatically deriving the self.attributes which are of instance type Hyperparameter
        hyperparameters = [self.n_epochs, self.learning_rate, self.classifier_dropout, self.warmup_step_fraction, self.use_gradient_clipping]
        hyperparameter_dictionary = {}
        for hyperparameter in hyperparameters:
            # for the first optimization trial use default values defined in the yaml configuration file
            if trial.number == 0:
                hyperparameter_dictionary[hyperparameter.name] = hyperparameter.default
            # for the remaining trials sample hyperparameters according to their definition in the yaml configuration file and their data type
            else:
                if hyperparameter.dtype == float:
                    hyperparameter_dictionary[hyperparameter.name] = trial.suggest_float(hyperparameter.name, hyperparameter.low, hyperparameter.high)
                elif hyperparameter.dtype == bool:
                    hyperparameter_dictionary[hyperparameter.name] = trial.suggest_categorical(hyperparameter.name, [True, False])
                elif hyperparameter.dtype == int:
                    hyperparameter_dictionary[hyperparameter.name] = trial.suggest_int(hyperparameter.name, hyperparameter.low, hyperparameter.high)
                else:
                    raise ValueError("Data type of the hyperparameters must be one of float, int or bool")
        
        # for a given set of hyperparameters, conduct cross validation for the training data
        cross_validation_score = self.cross_validate(
            n_folds = self.n_splits,
            training_data = self.training_data,
            training_batch_size = self.batch_size,
            validation_batch_size = self.batch_size,
            n_epochs = hyperparameter_dictionary["n_epochs"],
            learning_rate = hyperparameter_dictionary["learning_rate"],
            classifier_dropout = hyperparameter_dictionary["classifier_dropout"],
            warmup_step_fraction = hyperparameter_dictionary["warmup_step_fraction"],
            use_gradient_clipping = hyperparameter_dictionary["use_gradient_clipping"]
        )

        return cross_validation_score
    
    def cross_validate(
            self, 
            n_folds, 
            training_data, 
            training_batch_size, 
            validation_batch_size, 
            n_epochs, 
            learning_rate, 
            classifier_dropout, 
            warmup_step_fraction, 
            use_gradient_clipping
        ):

        """
        Performs cross-validation on the training data and returns the average score across folds.

        Parameters
        ----------
        n_folds : int
            The number of cross-validation folds.
        training_data : Dataset
            The training dataset to be used.
        training_batch_size : int
            The batch size for training.
        validation_batch_size : int
            The batch size for validation.
        n_epochs : int
            The number of epochs to train for.
        learning_rate : float
            The learning rate for the optimizer.
        classifier_dropout : float
            The dropout rate for the classifier layer.
        warmup_step_fraction : float
            The fraction of steps for learning rate warm-up.
        use_gradient_clipping : bool
            Whether to apply gradient clipping.

        Returns
        -------
        float
            The average cross-validation score across all folds.
        """

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        folder = KFold(n_splits=n_folds)
        split_scores = []
        for split, (training_index, validation_index) in enumerate(folder.split(training_data)):
            self.logger.info("-" * 100)
            self.logger.info(f"Starting training for split {split + 1}")
            training_split = training_data.select(training_index)
            validation_split = training_data.select(validation_index)

            self.logger.info(f"Counting occurences of labels...")
            self.logger.info(f"Occurence of labels for training data: {training_split.num_labels()}")
            self.logger.info(f"Occurence of labels for test data: {validation_split.num_labels()}")

            training_split = DataLoader(training_split, batch_size = training_batch_size, shuffle = False, collate_fn=self.collate_fn)
            validation_split = DataLoader(validation_split, batch_size = validation_batch_size, shuffle = False, collate_fn=self.collate_fn)

            split_score = self.train(training_split, validation_split, n_epochs, learning_rate, classifier_dropout, warmup_step_fraction, use_gradient_clipping, os.path.join(self.save_path, f"current_split_model"))
            self.logger.info(f"Split {split + 1} is finished, the score is: {split_score:.4f}")
            self.logger.info("-" * 100)
            split_scores.append(split_score)

        return np.mean(split_scores)

    def train(self, training_data, validation_data, n_epochs, learning_rate, classifier_dropout, warmup_step_fraction, use_gradient_clipping, save_best_model_path):
        
        """
        Trains the model on the training data and validates it on the validation data, with early stopping and checkpointing.

        Parameters
        ----------
        training_data : DataLoader
            The training data loader.
        validation_data : DataLoader
            The validation data loader.
        n_epochs : int
            The number of epochs to train for.
        learning_rate : float
            The learning rate for the optimizer.
        classifier_dropout : float
            The dropout rate for the classifier layer.
        warmup_step_fraction : float
            The fraction of steps for learning rate warm-up.
        use_gradient_clipping : bool
            Whether to apply gradient clipping.
        save_best_model_path : str
            The path to save the best model during training.

        Returns
        -------
        float
            The best validation score achieved during training.
        """

        # Early stopping instance with specified save path
        # during training the best model is saved temporarily as a checkpoint.pth file in the self.save_path/current_split_model folder
        # if a model has a lower loss during training in comparison to the loss of the last episode or if its performance was better
        # before early stopping, than, its parameters are reloaded from the best state saved in self.save_path/current_split_model/checkpoint.pth
        # I am not sure, if this is really the best way to do this, by manual checks if often happens that the score from the cross validation
        # would be higher if we do not reload the previous model, maybe one should also use the validation score to be optimized for early stopping and not 
        # the validation loss
        early_stopping = EarlyStopping(patience = self.early_stopping_patience, verbose = True, mode = 'min', save_path = save_best_model_path)

        # this initializes a pretrained model with new parameters for the classification head
        model = self._load_model(classifier_dropout)
        model.to(self.device)

        # warmup_step_fraction is the part of all iterations with a warm-up learning rate, it is a hyperparameter
        n_warmup = int(n_epochs * len(training_data) * warmup_step_fraction)
        # the learning rate is currently also specified as a hyperparameter
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps = n_epochs * len(training_data), num_warmup_steps = n_warmup)

        iteration = 0
        for epoch in range(n_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}")
            training_predictions, training_labels = [], []
            training_loss = 0
            for batch in training_data:
                inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                # if we want to train a model with an aggregation head, we need sequence_mask values in addition
                # which flag sequences in a document to 1 and padded sequence embeddings to 0
                if self.aggregated_document_model:
                    sequence_mask = batch["sequence_mask"].to(self.device)
                    model_output = model(input_ids = inputs, attention_mask = attention_mask, sequence_mask = sequence_mask, labels = labels)
                else:
                    model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                loss = model_output.loss
                training_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                # hyperparameter
                if use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
                optimizer.step()
                lr_scheduler.step()

                if iteration % 20 == 0:
                    logging.info(f"Current training batch loss: {loss.item():.4f} in epoch {epoch + 1}")

                batch_predictions = model_output.logits.argmax(dim = 1)
                training_predictions.append(batch_predictions)
                training_labels.append(labels)
                iteration += 1

            # after training an epochs, determine training metrics
            training_loss /= len(training_data)
            self.logger.info(f"Epoch finished, average loss over training batches: {training_loss:.4f}")
            training_predictions = torch.cat(training_predictions, dim = 0)
            training_labels = torch.cat(training_labels, dim = 0)

            self.logger.info("-"*100)
            self.logger.info("Training metrics:")
            self.logger.info("-"*100)
            # the self._determine_scores function returns task specific metrics for regression, binary and multi-classification
            if self.device.type == "cuda":
                self._determine_scores(training_labels.cpu().numpy(), training_predictions.cpu().numpy())
            else:
                self._determine_scores(training_labels.numpy(), training_predictions.numpy())
            
            # determine metrics for validation data
            validation_predictions, validation_labels = [], []
            validation_loss = 0
            with torch.no_grad():
                for batch in validation_data:
                    inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                    if self.aggregated_document_model:
                        sequence_mask = batch["sequence_mask"].to(self.device)
                        model_output = model(input_ids = inputs, attention_mask = attention_mask, sequence_mask = sequence_mask, labels = labels)
                    else:
                        model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                    validation_loss += model_output.loss.item()
                    batch_predictions = model_output.logits.argmax(dim = 1)
                    validation_predictions.append(batch_predictions)
                    validation_labels.append(labels)
                
            validation_loss /= len(validation_data)
            self.logger.info(f"Average loss over validation batches: {validation_loss:.4f}")
            validation_predictions = torch.cat(validation_predictions, dim = 0)
            validation_labels = torch.cat(validation_labels, dim = 0)

            self.logger.info("-"*100)
            self.logger.info("Validation metrics:")
            self.logger.info("-"*100)
            if self.device.type == "cuda":
                validation_scores = self._determine_scores(validation_labels.cpu().numpy(), validation_predictions.cpu().numpy())
            else:
                validation_scores = self._determine_scores(validation_labels.numpy(), validation_predictions.numpy())
            max_score = validation_scores['max_score']

            # uptdate early stopping, if the model improved the improved model is saved temporarily, if not we increase the patience value 
            # until early stopping is triggered
            early_stopping(validation_loss, model)

            # this whole block loads the best model during training (according to its loss)
            # (early_stopping.best_score > -validatio_loss) indicates that during training a smaller loss has been found than in the last epoch
            if early_stopping.early_stop or (epoch == n_epochs - 1) and (early_stopping.best_score > -validation_loss):
                if early_stopping.early_stop:
                    self.logger.info("Early stopping, loading best model from before and determine score...")
                else:
                    self.logger.info("Last epoch reached, validation loss was better before, loading best model during training.")
                model = self._load_model(classifier_dropout, os.path.join(save_best_model_path, "checkpoint.pth"))
                model.to(self.device)

                validation_predictions, validation_labels = [], []
                validation_loss = 0
                with torch.no_grad():
                    for batch in validation_data:
                        inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                        if self.aggregated_document_model:
                            sequence_mask = batch["sequence_mask"].to(self.device)
                            model_output = model(input_ids = inputs, attention_mask = attention_mask, sequence_mask = sequence_mask, labels = labels)
                        else:
                            model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                        validation_loss += model_output.loss.item()
                        batch_predictions = model_output.logits.argmax(dim = 1)
                        validation_predictions.append(batch_predictions)
                        validation_labels.append(labels)
                validation_loss /= len(validation_data)
                validation_predictions = torch.cat(validation_predictions, dim = 0)
                validation_labels = torch.cat(validation_labels, dim = 0)

                self.logger.info("-"*100)
                self.logger.info("Validation metrics after reloading the model before ending this training:")
                self.logger.info("-"*100)

                if self.device.type == "cuda":
                    validation_scores = self._determine_scores(validation_labels.cpu().numpy(), validation_predictions.cpu().numpy())
                else:
                    validation_scores = self._determine_scores(validation_labels.numpy(), validation_predictions.numpy())
                max_score = validation_scores['max_score']
                self.logger.info("Determined score from best model, ending training.")

                if early_stopping.early_stop:
                    break

        return max_score
    
    def final_evaluation(self, finetuned_model_path, classifier_dropout):
        
        """
        Evaluates the final trained model on both the training and test datasets, returning performance metrics.

        Parameters
        ----------
        finetuned_model_path : str
            The path to the fine-tuned model for evaluation.

        Returns
        -------
        Tuple[Dict[str, Any], Dict[str, Any]]
            The training and test performance metrics.
        """

        # it is important to understand that this is called after hyperparameter optimization and training the model on the full
        # training data, once this is finished, the final model is saved to the finetuned path from were the model_loader imports 
        # the final model here to evaluate it for one more time for training and test data
        model = self.model_loader(model_path = finetuned_model_path, num_labels = self.num_labels, classifier_dropout = classifier_dropout) 
        self.logger.info(f"Final model from {finetuned_model_path} is loaded.")
        model.to(self.device)

        training_data = DataLoader(self.training_data, self.batch_size, False, collate_fn=self.collate_fn)
        test_data = DataLoader(self.test_data, self.batch_size, False, collate_fn=self.collate_fn)

        self.logger.info("Determining training scores of final model.")
        with torch.no_grad():
            training_predictions, training_labels = [], []
            for batch in training_data:
                inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                if self.aggregated_document_model:
                    sequence_mask = batch["sequence_mask"].to(self.device)
                    model_output = model(input_ids = inputs, attention_mask = attention_mask, sequence_mask = sequence_mask, labels = labels)
                else:
                    model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                batch_predictions = model_output.logits.argmax(dim = 1)
                training_predictions.append(batch_predictions)
                training_labels.append(labels)

        training_predictions = torch.cat(training_predictions, dim = 0)
        training_labels = torch.cat(training_labels, dim = 0)
        if self.device.type == "cuda":
            training_scores = self._determine_scores(training_labels.cpu().numpy(), training_predictions.cpu().numpy())
        else:
            training_scores = self._determine_scores(training_labels.numpy(), training_predictions.numpy())

        self.logger.info("Determining test scores of final model.")
        test_predictions, test_labels = [], []
        with torch.no_grad():
            for batch in test_data:
                inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                if self.aggregated_document_model:
                    sequence_mask = batch["sequence_mask"].to(self.device)
                    model_output = model(input_ids = inputs, attention_mask = attention_mask, sequence_mask = sequence_mask, labels = labels)
                else:
                    model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                batch_predictions = model_output.logits.argmax(dim = 1)
                test_predictions.append(batch_predictions)
                test_labels.append(labels)
            
        test_predictions = torch.cat(test_predictions, dim = 0)
        test_labels = torch.cat(test_labels, dim = 0)
        if self.device.type == "cuda":
            test_scores = self._determine_scores(test_labels.cpu().numpy(), test_predictions.cpu().numpy())
        else:
            test_scores = self._determine_scores(test_labels.numpy(), test_predictions.numpy())
        return training_scores, test_scores


    def _determine_scores(self, true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, Any]:

        """
        Computes and logs performance metrics based on the true and predicted labels.

        Parameters
        ----------
        true_labels : np.ndarray
            The true labels for the data.
        predicted_labels : np.ndarray
            The predicted labels from the model.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing performance metrics such as accuracy, precision, recall, and F1 score.
        """

        if self.task == "regression":
            mae = mean_absolute_error(true_labels, predicted_labels)
            mse = mean_squared_error(true_labels, predicted_labels)
            r2 = r2_score(true_labels, predicted_labels)
            
            scores = dict(
                mean_absolute_error = mae,
                mean_squared_error = mse,
                max_score = r2,
            )

            self.logger.info(f"Mean absolute error: {scores['mean_absolute_error']:.4f}")
            self.logger.info(f"Mean squared error: {scores['mean_squared_error']:.4f}")
            self.logger.info(f"R2 score: {scores['max_score']:.4f}")

        elif self.task == "binary_classification":
            accuracy_scores = accuracy_score(true_labels, predicted_labels)
            precision_scores = precision_score(true_labels, predicted_labels)
            recall_scores = recall_score(true_labels, predicted_labels)
            f1_scores = f1_score(true_labels, predicted_labels)

            scores = dict(
                accuracy_score = accuracy_scores,
                precision_score = precision_scores,
                recall_score = recall_scores,
                max_score = f1_scores
            )

            self.logger.info(f"Accuracy: {scores['accuracy_score']:.4f}")
            self.logger.info(f"Precision: {scores['precision_score']:.4f}")
            self.logger.info(f"Recall: {scores['recall_score']:.4f}")
            self.logger.info(f"F1 score: {scores['max_score']:.4f}")

        else:
            accuracy_scores = accuracy_score(true_labels, predicted_labels)
            precision_scores = precision_score(true_labels, predicted_labels, average = None, zero_division = 0)
            recall_scores = recall_score(true_labels, predicted_labels, average = None, zero_division = 0)
            average_f1_score = f1_score(true_labels, predicted_labels, average = "macro", zero_division = 0)

            scores = dict(
                accuracy_score = accuracy_scores,
                precision_scores = precision_scores,
                recall_scores = recall_scores,
                max_score = average_f1_score
            )

            self.logger.info(f"Accuracy: {scores['accuracy_score']:.4f}")
            for label_class in range(len(scores["precision_scores"])):
                self.logger.info(f"Precision score for label_class {label_class}: {scores['precision_scores'][label_class]:.4f}")
            for label_class in range(len(scores["recall_scores"])):
                self.logger.info(f"Recall score for label_class {label_class}: {scores['recall_scores'][label_class]:.4f}")
            self.logger.info(f"Average F1 score: {scores['max_score']:.4f}") 

        return scores

    def _load_model(self, classifier_dropout: float, save_path = None):

        """
        Loads an encoder model for sequence classification with a specified dropout rate.

        This method loads a pre-trained model for sequence classification, applying the specified 
        dropout rate to the classifier layer. If a save path is provided, the model's state is loaded from 
        the specified checkpoint.

        Parameters
        ----------
        classifier_dropout : float
            The dropout rate to apply to the classifier layer.
        save_path : str, optional
            The path to a saved model checkpoint to load. If None, the model is loaded without applying any checkpoint (default is None).

        Returns
        -------
        ElectraForSequenceClassification
            The loaded ELECTRA model, ready for training or evaluation.
        """

        # by calling the model_loader and use the self.model_path, the pretrained model gets imported with the classifier_dropout which is a hyperparameter
        model = self.model_loader(self.model_path, num_labels = self.num_labels, classifier_dropout = classifier_dropout)
        # the save_path option is used only during training if the best model over all epochs is imported at the end of all epochs or after early stopping
        if save_path:
            self.logger.info(f"Loading model from {save_path}")
            model.load_state_dict(torch.load(save_path))

        return model       

