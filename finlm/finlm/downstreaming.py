import optuna
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from transformers import ElectraForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from finlm.dataset import FinetuningDataset
from finlm.callbacks import EarlyStopping
from typing import Dict, Any, Optional
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Hyperparameter:
    def __init__(self, name, dtype, low = None, high = None, default = None):
        self.name = name
        self.dtype = dtype
        self.low = low
        self.high = high
        self.default = default

class EncoderClassificationDownstreaming:
    def __init__(
            self,
            model_path: str,
            num_labels: int,
            device: torch.device,
            tokenizer_path: str,
            max_sequence_length: int,
            dataset: Dataset,
            text_column: str,
            dataset_columns: list[str],
            shuffle_data: bool = True,
            shuffle_data_random_seed: Optional[int] = None,
            training_data_fraction: float = 0.80,
            n_epochs: Hyperparameter = None,
            learning_rate: Hyperparameter = None,
            classifier_dropout: Hyperparameter = None,
            warmup_step_fraction: Hyperparameter = None,
            use_gradient_clipping: Hyperparameter = None,
            save_path: str = None
        ):
    
        self.model_path = model_path
        self.num_labels = num_labels
        self.device = device
        self.dataset = FinetuningDataset(
            tokenizer_path,
            max_sequence_length,
            dataset,
            text_column,
            dataset_columns,
            shuffle_data,
            shuffle_data_random_seed,
            training_data_fraction
        )
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.classifier_dropout = classifier_dropout
        self.warmup_step_fraction = warmup_step_fraction
        self.use_gradient_clipping = use_gradient_clipping
        self.save_path = save_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Occurence of labels for training data: {np.unique(self.dataset.training_data['label'], return_counts=True)}")
        self.logger.info(f"Occurence of labels for test data: {np.unique(self.dataset.test_data['label'], return_counts=True)}")

    def train_optuna_optimized_cv_model(self, n_trials):
        best_params, _ = self.optuna_optimize(n_trials = n_trials)
        full_training_split = DataLoader(self.dataset.training_data, batch_size = 32, shuffle = False)
        test_split = DataLoader(self.dataset.test_data, batch_size = 32, shuffle = False)

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

        model = ElectraForSequenceClassification.from_pretrained(self.model_path, num_labels = self.num_labels)        
        self.logger.info(f"Loading model finetuned model from checkpoint.")
        model.load_state_dict(torch.load(os.path.join(self.save_path, "checkpoint.pth")))
        model.save_pretrained(os.path.join(self.save_path, "finetuned_model"))
        self.model_and_token_info_to_json()

    def optuna_optimize(self, n_trials = 10):
            study = optuna.create_study(direction = "maximize")
            study.optimize(self.optuna_objective, n_trials = n_trials)            
            return study.best_params, study.best_value

    def optuna_objective(self, trial):
        hyperparameters = [self.n_epochs, self.learning_rate, self.classifier_dropout, self.warmup_step_fraction, self.use_gradient_clipping]
        hyperparameter_dictionary = {}
        for hyperparameter in hyperparameters:
            if trial.number == 0:
                hyperparameter_dictionary[hyperparameter.name] = hyperparameter.default
            else:
                if hyperparameter.dtype == float:
                    hyperparameter_dictionary[hyperparameter.name] = trial.suggest_float(hyperparameter.name, hyperparameter.low, hyperparameter.high)
                elif hyperparameter.dtype == bool:
                    hyperparameter_dictionary[hyperparameter.name] = trial.suggest_categorical(hyperparameter.name, [True, False])
                elif hyperparameter.dtype == int:
                    hyperparameter_dictionary[hyperparameter.name] = trial.suggest_int(hyperparameter.name, hyperparameter.low, hyperparameter.high)
                else:
                    raise ValueError("Data type of the hyperparameters must be one of float, int or bool")
        
        cross_validation_score = self.cross_validate(
            n_folds = 5,
            training_data = self.dataset.training_data,
            training_batch_size = 32,
            validation_batch_size = 32,
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

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        folder = KFold(n_splits=n_folds)
        split_scores = []
        for split, (training_index, validation_index) in enumerate(folder.split(training_data)):
            self.logger.info("-" * 100)
            self.logger.info(f"Starting training for split {split + 1}")
            training_split = training_data.select(training_index)
            validation_split = training_data.select(validation_index)

            self.logger.info(f"Occurence of labels for training split: {np.unique(training_split['label'], return_counts=True)}")
            self.logger.info(f"Occurence of labels for validation split: {np.unique(validation_split['label'], return_counts=True)}")

            training_split = DataLoader(training_split, batch_size = training_batch_size, shuffle = False)
            validation_split = DataLoader(validation_split, batch_size = validation_batch_size, shuffle = False)

            split_score = self.train(training_split, validation_split, n_epochs, learning_rate, classifier_dropout, warmup_step_fraction, use_gradient_clipping, os.path.join(self.save_path, f"current_split_model"))
            self.logger.info(f"Split {split + 1} is finished, the score is: {split_score:.4f}")
            self.logger.info("-" * 100)
            split_scores.append(split_score)

        return np.mean(split_scores)

    def train(self, training_data, validation_data, n_epochs, learning_rate, classifier_dropout, warmup_step_fraction, use_gradient_clipping, save_best_model_path):
        
        # Early stopping instance with specified save path
        early_stopping = EarlyStopping(patience = 2, verbose = True, mode = 'min', save_path = save_best_model_path)

        model = self._load_electra_model(classifier_dropout)
        model.to(self.device)

        n_warmup = int(n_epochs * len(training_data) * warmup_step_fraction)
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps = n_epochs * len(training_data), num_warmup_steps = n_warmup)

        iteration = 0
        for epoch in range(n_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}")
            training_predictions, training_labels = [], []
            training_loss = 0
            for batch in training_data:
                inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                loss = model_output.loss
                training_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
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

            training_loss /= len(training_data)
            self.logger.info(f"Epoch finished, average loss over training batches: {training_loss:.4f}")
            training_predictions = torch.cat(training_predictions, dim = 0)
            training_labels = torch.cat(training_labels, dim = 0)
            if self.device.type == "cuda":
                training_scores = self._determine_multi_classification_scores(training_labels.cpu().numpy(), training_predictions.cpu().numpy())
            else:
                training_scores = self._determine_multi_classification_scores(training_labels.numpy(), training_predictions.numpy())
            self.logger.info(f"After iteration {iteration} in epoch {epoch + 1} the training scores are:")
            self.logger.info("-" * 100)
            self._logging_multi_classification_scores(training_scores)
            self.logger.info("-" * 100)
            
            validation_predictions, validation_labels = [], []
            validation_loss = 0
            with torch.no_grad():
                for batch in validation_data:
                    inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                    model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                    validation_loss += model_output.loss.item()
                    batch_predictions = model_output.logits.argmax(dim = 1)
                    validation_predictions.append(batch_predictions)
                    validation_labels.append(labels)
                
            validation_loss /= len(validation_data)
            self.logger.info(f"Average loss over validation batches: {validation_loss:.4f}")
            validation_predictions = torch.cat(validation_predictions, dim = 0)
            validation_labels = torch.cat(validation_labels, dim = 0)
            if self.device.type == "cuda":
                validation_scores = self._determine_multi_classification_scores(validation_labels.cpu().numpy(), validation_predictions.cpu().numpy())
            else:
                validation_scores = self._determine_multi_classification_scores(validation_labels.numpy(), validation_predictions.numpy())
            validation_f1_score = validation_scores['average_f1_score']
            self.logger.info(f"After iteration {iteration} in epoch {epoch + 1} the validation scores are:")
            self.logger.info("-" * 100)
            self._logging_multi_classification_scores(validation_scores)
            self.logger.info("-" * 100)

            early_stopping(validation_loss, model)

            if early_stopping.early_stop or (epoch == n_epochs - 1) and (early_stopping.best_score > -validation_loss):
                if early_stopping.early_stop:
                    self.logger.info("Early stopping, loading best model from before and determine score...")
                else:
                    self.logger.info("Last epoch reached, validation loss was better before, loading best model during training.")
                model = self._load_electra_model(classifier_dropout, os.path.join(save_best_model_path, "checkpoint.pth"))
                model.to(self.device)

                validation_predictions, validation_labels = [], []
                validation_loss = 0
                with torch.no_grad():
                    for batch in validation_data:
                        inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                        model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                        validation_loss += model_output.loss.item()
                        batch_predictions = model_output.logits.argmax(dim = 1)
                        validation_predictions.append(batch_predictions)
                        validation_labels.append(labels)
                validation_loss /= len(validation_data)
                validation_predictions = torch.cat(validation_predictions, dim = 0)
                validation_labels = torch.cat(validation_labels, dim = 0)
                if self.device.type == "cuda":
                    validation_scores = self._determine_multi_classification_scores(validation_labels.cpu().numpy(), validation_predictions.cpu().numpy())
                else:
                    validation_scores = self._determine_multi_classification_scores(validation_labels.numpy(), validation_predictions.numpy())
                validation_f1_score = validation_scores['average_f1_score']
                self.logger.info("Determined score from best model, ending training.")

                if early_stopping.early_stop:
                    break

        return validation_f1_score
    
    def final_evaluation(self, fintuned_model_path):
        
        model = ElectraForSequenceClassification.from_pretrained(fintuned_model_path)
        model.to(self.device)

        training_data = DataLoader(self.dataset.training_data, 32, False)
        test_data = DataLoader(self.dataset.test_data, 32, False)

        self.logger.info("Determining training scores of final model.")
        with torch.no_grad():
            training_predictions, training_labels = [], []
            for batch in training_data:
                inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                batch_predictions = model_output.logits.argmax(dim = 1)
                training_predictions.append(batch_predictions)
                training_labels.append(labels)

        training_predictions = torch.cat(training_predictions, dim = 0)
        training_labels = torch.cat(training_labels, dim = 0)
        if self.device.type == "cuda":
            training_scores = self._determine_multi_classification_scores(training_labels.cpu().numpy(), training_predictions.cpu().numpy())
        else:
            training_scores = self._determine_multi_classification_scores(training_labels.numpy(), training_predictions.numpy())

        self.logger.info("Determining test scores of final model.")
        test_predictions, test_labels = [], []
        with torch.no_grad():
            for batch in test_data:
                inputs, attention_mask, labels = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device), batch["label"].to(self.device)
                model_output = model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                batch_predictions = model_output.logits.argmax(dim = 1)
                test_predictions.append(batch_predictions)
                test_labels.append(labels)
            
        test_predictions = torch.cat(test_predictions, dim = 0)
        test_labels = torch.cat(test_labels, dim = 0)
        if self.device.type == "cuda":
            test_scores = self._determine_multi_classification_scores(test_labels.cpu().numpy(), test_predictions.cpu().numpy())
        else:
            test_scores = self._determine_multi_classification_scores(test_labels.numpy(), test_predictions.numpy())
        return training_scores, test_scores
    
    @staticmethod
    def _determine_multi_classification_scores(true_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, float]: 
        accuracy_scores = accuracy_score(true_labels, predicted_labels)
        precision_scores = precision_score(true_labels, predicted_labels, average = None, zero_division = 0)
        recall_scores = recall_score(true_labels, predicted_labels, average = None, zero_division = 0)
        average_f1_score = f1_score(true_labels, predicted_labels, average = "macro", zero_division = 0)

        scores = dict(
            accuracy_score = accuracy_scores,
            precision_scores = precision_scores,
            recall_scores = recall_scores,
            average_f1_score = average_f1_score
        )

        return scores
    
    def _logging_multi_classification_scores(self, scores):
        self.logger.info(f"Accuracy: {scores['accuracy_score']:.4f}")
        for label_class in range(len(scores["precision_scores"])):
            self.logger.info(f"Precision score for label_class {label_class}: {scores['precision_scores'][label_class]:.4f}")
        for label_class in range(len(scores["recall_scores"])):
            self.logger.info(f"Recall score for label_class {label_class}: {scores['recall_scores'][label_class]:.4f}")
        self.logger.info(f"Average F1 score: {scores['average_f1_score']:.4f}") 

    def _load_electra_model(self, classifier_dropout, save_path = None):
        model = ElectraForSequenceClassification.from_pretrained(self.model_path, num_labels = self.num_labels, classifier_dropout = classifier_dropout)        
        if save_path:
            self.logger.info(f"Loading model from {save_path}")
            model.load_state_dict(torch.load(save_path))

        return model       

    def model_and_token_info_to_json(self):
        metadata = {}
        metadata["model_path"] = self.model_path
        metadata["tokenizer_path"] = self.dataset.tokenizer_path
        with open(os.path.join(self.save_path, "tokenizer_and_model_info.json"), "w") as file:
            json.dump(metadata, file, indent = 4)