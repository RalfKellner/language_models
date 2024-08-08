import os
from dataclasses import asdict
from finlm.dataset import FinLMDataset
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, ElectraPreTrainedModel, ElectraModel
from transformers import get_linear_schedule_with_warmup
from finlm.config import FinLMConfig
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import torch
from torcheval.metrics.functional import binary_precision, binary_recall
from torch import nn
from typing import Optional, Any
import math
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#########################################################################################################################
# Models for pretraining
#########################################################################################################################

class PretrainLM:
    def __init__(self, config: FinLMConfig):
        self.config = config
        self.dataset_config = self.config.dataset_config
        self.model_config = self.config.model_config
        self.optimization_config = self.config.optimization_config
        self.save_root_path = config.save_models_and_results_to
        self.logger = logging.getLogger(self.__class__.__name__)
        self._set_device()

    def load_dataset(self):
        self.dataset = FinLMDataset.from_dict(asdict(self.dataset_config))

    @staticmethod
    def mask_tokens(inputs, mlm_probability, mask_token_id, special_token_ids, n_tokens, ignore_index = -100, hard_masking = False):
        device = inputs.device
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
        # create special_token_mask, first set all entries to false
        special_tokens_mask = torch.full(labels.shape, False, dtype = torch.bool, device = device)
        # flag all special tokens as true
        for sp_id in special_token_ids:
            special_tokens_mask = special_tokens_mask | (inputs == sp_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if ignore_index:
            labels[~masked_indices] = ignore_index  # We only compute loss on masked tokens

        if hard_masking:
            inputs[masked_indices] = mask_token_id
        else:
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device = device)).bool() & masked_indices
            inputs[indices_replaced] = mask_token_id 

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device = device)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(n_tokens, labels.shape, dtype=torch.long, device = device)
            inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def _create_directory_and_return_save_path(self, model_type):
        current_model_folder_paths = os.listdir(self.save_root_path)
        current_model_type_folder_names = [model for model in current_model_folder_paths if model.startswith(model_type)]
        if len(current_model_type_folder_names) > 0:
            current_model_type_index = max([int(model_name.split("_")[1]) for model_name in current_model_type_folder_names])
            new_model_path = self.save_root_path + model_type + "_" + str(current_model_type_index + 1).zfill(2) + "/"
        else:
            new_model_path = self.save_root_path + model_type + "_00/"
        os.mkdir(new_model_path)
        return new_model_path

    def _set_device(self):
        if not(torch.cuda.is_available()):
            logging.warning("GPU seems to be unavailable.")
        else:
            self.device = torch.device("cuda")


class PretrainMLM(PretrainLM):
    def __init__(self, config):
        super().__init__(config)
        self.prepare_data_model_optimizer()

    def load_model(self):
        self.model_config = ElectraConfig(
            vocab_size = self.dataset.tokenizer.vocab_size,
            embedding_size = self.model_config.embedding_size,
            hidden_size = self.model_config.hidden_size, 
            num_hidden_layers = self.model_config.num_hidden_layers,
            num_attention_heads = self.model_config.num_attention_heads,
            intermediate_size = self.model_config.intermediate_size
        )

        self.model = ElectraForMaskedLM(self.model_config)
        self.model.to(self.device)

    def load_optimization(self):

        n_sequences = 0
        for key in self.dataset.database_retrieval.keys():
            n_sequences += self.dataset.database_retrieval[key]["limit"]

        self.iteration_steps_per_epoch = int(np.ceil(n_sequences / self.dataset.batch_size))
        total_steps = self.iteration_steps_per_epoch * self.optimization_config.n_epochs  
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.optimization_config.learning_rate) 
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = self.optimization_config.lr_scheduler_warm_up_steps, num_training_steps = total_steps)

    def prepare_data_model_optimizer(self):
        self.load_dataset()
        self.load_model()
        self.load_optimization()

    def train(self):

        self.logger.info("Starting with training...")
        training_metrics = {}
        training_metrics["loss"] = []
        training_metrics["accuracy"] = []
        training_metrics["gradient_norms"] = []
        training_metrics["learning_rates"] = []

        for epoch in range(self.optimization_config.n_epochs): 

            # update the offset for database retrieval, epoch = 0 -> offset = 0, epoch = 1 -> offset = 1 * limit, epoch = 2 -> offset = 2 * limit, ...    
            self.dataset.set_dataset_offsets(epoch)
            self.dataset.prepare_data_loader()

            for batch_id, batch in enumerate(self.dataset):
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                inputs, labels = self.mask_tokens(
                    inputs,
                    mlm_probability = self.optimization_config.mlm_probability,
                    mask_token_id = self.dataset.mask_token_id,
                    special_token_ids = self.dataset.special_token_ids,
                    n_tokens = self.dataset.tokenizer.vocab_size)

                mlm_output = self.model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                mlm_loss, mlm_logits = mlm_output.loss, mlm_output.logits
                training_metrics["loss"].append(mlm_loss.item())

                # gradient determination and update
                self.optimizer.zero_grad()

                # determine gradients
                mlm_loss.backward()

                if self.optimization_config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

                # determine gradient norms, equal to one if use_gradient_clipping is set to True
                mlm_grads = [p.grad.detach().flatten() for p in self.model.parameters()]
                mlm_grad_norm = torch.cat(mlm_grads).norm()

                # update parameters        
                self.optimizer.step()
                # update learning rate
                self.scheduler.step()

                # determine accuracy metrics, (maybe check for correctness later, has been implemented quickly;))
                with torch.no_grad():
                    # mask to identify ids which have been masked before
                    masked_ids_mask = inputs == self.dataset.tokenizer.mask_token_id
                    predictions = mlm_logits.argmax(-1)
                    mlm_accuracy = (predictions[masked_ids_mask] == labels[masked_ids_mask]).float().mean()

                training_metrics["accuracy"].append(mlm_accuracy.item())
                training_metrics["gradient_norms"].append(mlm_grad_norm.item())
                current_lr = self.scheduler.get_last_lr()[0]
                training_metrics["learning_rates"].append(current_lr)

                if batch_id % 100 == 0:
                    self.logger.info(f"Results after {batch_id/self.iteration_steps_per_epoch:.4%} iterations of epoch {epoch+1}:")
                    self.logger.info(f"MLM loss: {mlm_loss.item():.4f}")
                    self.logger.info(f"Gradient norm: {mlm_grad_norm:.4f}")
                    self.logger.info(f"Current learning rate: {current_lr}")
                    self.logger.info(f"Accuracy for masking task: {mlm_accuracy.item():.4f}")
                    self.logger.info("-"*100)   

        self.logger.info("...training is finished, saving results and model.")

        save_path = self._create_directory_and_return_save_path(model_type = "mlm")
        training_metrics_df = pd.DataFrame(training_metrics)
        training_metrics_df.to_csv(save_path + "training_metrics.csv", index = False)
        training_metrics_df.loc[:, ["loss"]].plot()
        plt.savefig(save_path + "loss.png")
        training_metrics_df.loc[:, ["accuracy"]].plot()
        plt.savefig(save_path + "accuracy.png")
        self.model.save_pretrained(save_path + "mlm_model")
        self.config.to_json(save_path + "model_config.json")

        self.logger.info("Results and model are saved.")


class PretrainDiscriminator(PretrainLM):
    def __init__(self, config):
        super().__init__(config)
        self.prepare_data_model_optimizer()

    def load_model(self):
        self.model_config = ElectraConfig(
            vocab_size = self.dataset.tokenizer.vocab_size,
            embedding_size = self.model_config.embedding_size,
            hidden_size = self.model_config.hidden_size, 
            num_hidden_layers = self.model_config.num_hidden_layers,
            num_attention_heads = self.model_config.num_attention_heads
        )

        self.model = ElectraForPreTraining(self.model_config)
        self.model.to(self.device)

    def load_optimization(self):

        n_sequences = 0
        for key in self.dataset.database_retrieval.keys():
            n_sequences += self.dataset.database_retrieval[key]["limit"]

        self.iteration_steps_per_epoch = int(np.ceil(n_sequences / self.dataset.batch_size))
        total_steps = self.iteration_steps_per_epoch * self.optimization_config.n_epochs  
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.optimization_config.learning_rate) 
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = self.optimization_config.lr_scheduler_warm_up_steps, num_training_steps = total_steps)

    def prepare_data_model_optimizer(self):
        self.load_dataset()
        self.load_model()
        self.load_optimization()


    def replace_masked_tokens_randomly(self, inputs, mlm_probability, mask_token_id, special_token_ids, n_tokens, hard_masking = True): 
        device = inputs.device
        masked_inputs, original_inputs = self.mask_tokens(
            inputs = inputs,
            mlm_probability = mlm_probability,
            mask_token_id = mask_token_id,
            special_token_ids = special_token_ids,
            n_tokens = n_tokens,
            ignore_index = None,
            hard_masking = hard_masking
            )
        
        masked_indices = torch.where(masked_inputs == mask_token_id, True, False)
        random_words = torch.randint(n_tokens, original_inputs.shape, dtype=torch.long, device = device)
        corrupted_inputs = original_inputs.clone()
        corrupted_inputs[masked_indices] = random_words[masked_indices]
        labels = torch.full(corrupted_inputs.shape, False, dtype=torch.bool, device=device)
        labels[masked_indices] = original_inputs[masked_indices] != corrupted_inputs[masked_indices]

        return corrupted_inputs, labels.float()

    def train(self):
        self.logger.info("Starting with training...")

        training_metrics = {}
        training_metrics["loss"] = []
        training_metrics["accuracy"] = []
        training_metrics["precision"] = []
        training_metrics["recall"] = []
        training_metrics["gradient_norms"] = []
        training_metrics["learning_rates"] = []

        for epoch in range(self.optimization_config.n_epochs): 

            # update the offset for database retrieval, epoch = 0 -> offset = 0, epoch = 1 -> offset = 1 * limit, epoch = 2 -> offset = 2 * limit, ...    
            self.dataset.set_dataset_offsets(epoch)
            self.dataset.prepare_data_loader()

            for batch_id, batch in enumerate(self.dataset):
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                inputs, labels = self.replace_masked_tokens_randomly(
                    inputs, 
                    mlm_probability = self.optimization_config.mlm_probability,
                    mask_token_id = self.dataset.mask_token_id,
                    special_token_ids = self.dataset.special_token_ids,
                    n_tokens = self.dataset.tokenizer.vocab_size,
                    hard_masking = True
                )

                discriminator_output = self.model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                discriminator_loss, discriminator_logits = discriminator_output.loss, discriminator_output.logits
                training_metrics["loss"].append(discriminator_loss.item())

                # gradient determination and update
                self.optimizer.zero_grad()

                # determine gradients
                discriminator_loss.backward()

                if self.optimization_config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

                # determine gradient norms, equal to one if use_gradient_clipping is set to True
                discriminator_grads = [p.grad.detach().flatten() for p in self.model.parameters()]
                discriminator_grad_norm = torch.cat(discriminator_grads).norm()

                # update parameters        
                self.optimizer.step()
                # update learning rate
                self.scheduler.step()

                # determine accuracy metrics, (maybe check for correctness later, has been implemented quickly;))
                with torch.no_grad():
                    active_loss = attention_mask == 1
                    active_logits = discriminator_logits[active_loss]
                    active_predictions = (torch.sign(active_logits) + 1.0) * 0.5
                    active_labels = labels[active_loss]

                    discriminator_accuracy = (active_predictions == active_labels).float().mean()
                    discriminator_precision = binary_precision(active_predictions.long(), active_labels.long())
                    discriminator_recall = binary_recall(active_predictions.long(), active_labels.long())

                training_metrics["accuracy"].append(discriminator_accuracy.item())
                training_metrics["precision"].append(discriminator_precision.item())
                training_metrics["recall"].append(discriminator_recall.item())
                training_metrics["gradient_norms"].append(discriminator_grad_norm.item())
                current_lr = self.scheduler.get_last_lr()[0]
                training_metrics["learning_rates"].append(current_lr)

                if batch_id % 100 == 0:
                    self.logger.info(f"Results after {batch_id/self.iteration_steps_per_epoch:.4%} iterations of epoch {epoch+1}:")
                    self.logger.info(f"Discriminator loss: {discriminator_loss.item():.4f}")
                    self.logger.info(f"Gradient norm: {discriminator_grad_norm:.4f}")
                    self.logger.info(f"Current learning rate: {current_lr}")
                    self.logger.info(f"Accuracy for replacement task: {discriminator_accuracy.item():.4f}")
                    self.logger.info(f"Precision for replacement task: {discriminator_precision.item():.4f}")
                    self.logger.info(f"Recall for replacement task: {discriminator_recall.item():.4f}")
                    self.logger.info("-"*100)   

        self.logger.info("...training is finished, saving results and model.")
        
        save_path = self._create_directory_and_return_save_path(model_type = "discriminator")
        training_metrics_df = pd.DataFrame(training_metrics)
        training_metrics_df.to_csv(save_path + "training_metrics.csv", index = False)
        training_metrics_df.loc[:, ["loss"]].plot()
        plt.savefig(save_path + "loss.png")
        training_metrics_df.loc[:, ["accuracy", "precision", "recall"]].plot(subplots = True)
        plt.savefig(save_path + "accuracy.png")
        self.model.save_pretrained(save_path + "discriminator_model")
        self.config.to_json(save_path + "model_config.json")

        self.logger.info("Results and model are saved.")
        

class PretrainElectra(PretrainLM):
    def __init__(self, config):
        super().__init__(config)
        self.prepare_data_model_optimizer()

    def load_model(self):
        self.generator_model_config = ElectraConfig(
            vocab_size = self.dataset.tokenizer.vocab_size,
            embedding_size = self.model_config.embedding_size,
            hidden_size = int(self.model_config.hidden_size * self.model_config.generator_size), 
            intermediate_size = int(self.model_config.intermediate_size * self.model_config.generator_size),
            num_hidden_layers = int(self.model_config.num_hidden_layers * self.model_config.generator_layer_size),
            num_attention_heads = int(self.model_config.num_attention_heads * self.model_config.generator_size)
        )

        self.discriminator_model_config = ElectraConfig(
            vocab_size = self.dataset.tokenizer.vocab_size,
            embedding_size = self.model_config.embedding_size,
            hidden_size = self.model_config.hidden_size, 
            num_hidden_layers = self.model_config.num_hidden_layers,
            num_attention_heads = self.model_config.num_attention_heads
        )

        self.generator = ElectraForMaskedLM(self.generator_model_config)
        self.discriminator = ElectraForPreTraining(self.discriminator_model_config)
        # tie word and position embeddings
        self.generator.electra.embeddings.word_embeddings = self.discriminator.electra.embeddings.word_embeddings
        self.generator.electra.embeddings.position_embeddings = self.discriminator.electra.embeddings.position_embeddings
        # add to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def load_optimization(self):

        # identify trainable parameters without duplicating the embedding and position parameters
        self.model_parameters = []
        # generator
        for name, params in self.discriminator.named_parameters():
            self.model_parameters.append(params)
        # discriminator
        for name, params in self.generator.named_parameters():
            if name.endswith("word_embeddings.weight") | name.endswith("position_embeddings.weight"):
                continue
            else:
                self.model_parameters.append(params)
        
        n_sequences = 0
        for key in self.dataset.database_retrieval.keys():
            n_sequences += self.dataset.database_retrieval[key]["limit"]
        self.iteration_steps_per_epoch = int(np.ceil(n_sequences / self.dataset.batch_size))
        total_steps = self.iteration_steps_per_epoch * self.optimization_config.n_epochs 
        self.optimizer = torch.optim.AdamW(self.model_parameters, lr = self.optimization_config.learning_rate) 
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = self.optimization_config.lr_scheduler_warm_up_steps, num_training_steps = total_steps)

    def prepare_data_model_optimizer(self):
        self.load_dataset()
        self.load_model()
        self.load_optimization()

    @staticmethod
    def replace_masked_tokens_from_generator(masked_inputs, original_inputs, logits, special_mask_id, discriminator_sampling = "multinomial"):
    
        device = masked_inputs.device
        discriminator_inputs = masked_inputs.clone()
        mask_indices = masked_inputs == special_mask_id

        if discriminator_sampling == "aggressive":
            sampled_ids = logits[mask_indices].argmax(-1)
        elif discriminator_sampling == "gumbel_softmax":
            sampled_ids = torch.nn.functional.gumbel_softmax(logits[mask_indices], hard = False).argmax(-1)
        else:
            sampled_ids = torch.multinomial(torch.nn.functional.softmax(logits[mask_indices], dim = -1), 1).squeeze()

        discriminator_inputs[mask_indices] = sampled_ids
        # initialize discriminator labels with False
        discriminator_labels = torch.full(masked_inputs.shape, False, dtype=torch.bool, device=device)
        # replace False with True if an id is sampled and not the same as the original one
        discriminator_labels[mask_indices] = discriminator_inputs[mask_indices] != original_inputs[mask_indices]
        # convert to float 
        discriminator_labels = discriminator_labels.float()

        return discriminator_inputs, discriminator_labels

    def train(self):

        self.logger.info("Starting with training...")
        training_metrics = {}
        training_metrics["loss"] = []
        training_metrics["mlm_loss"] = []
        training_metrics["discriminator_loss"] = []
        training_metrics["mlm_accuracy"] = []
        training_metrics["discriminator_accuracy"] = []
        training_metrics["discriminator_precision"] = []
        training_metrics["discriminator_recall"] = []
        training_metrics["gradient_norm"] = []
        training_metrics["learning_rates"] = []

        for epoch in range(self.optimization_config.n_epochs): 
            
            # update the offset for database retrieval, epoch = 0 -> offset = 0, epoch = 1 -> offset = 1 * limit, epoch = 2 -> offset = 2 * limit, ...    
            self.dataset.set_dataset_offsets(epoch)
            self.dataset.prepare_data_loader()

            for batch_id, batch in enumerate(self.dataset):
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                original_inputs = inputs.clone()
                generator_inputs, generator_labels = self.mask_tokens(
                    inputs,
                    mlm_probability = self.optimization_config.mlm_probability,
                    mask_token_id = self.dataset.mask_token_id,
                    special_token_ids = self.dataset.special_token_ids,
                    n_tokens = self.dataset.tokenizer.vocab_size)

                mlm_output = self.generator(input_ids = generator_inputs, attention_mask = attention_mask, labels = generator_labels)
                mlm_loss, mlm_logits = mlm_output.loss, mlm_output.logits

                sampling_logits = mlm_logits.detach()
                discriminator_inputs, discriminator_labels = self.replace_masked_tokens_from_generator(
                    masked_inputs = generator_inputs,
                    original_inputs = original_inputs,
                    logits = sampling_logits,
                    special_mask_id = self.dataset.tokenizer.mask_token_id,
                    discriminator_sampling = self.optimization_config.discriminator_sampling
                    )
                
                discriminator_output = self.discriminator(input_ids = discriminator_inputs, attention_mask = attention_mask, labels = discriminator_labels)
                discriminator_loss, discriminator_logits = discriminator_output.loss, discriminator_output.logits

                loss = mlm_loss + self.optimization_config.discriminator_weight * discriminator_loss

                training_metrics["loss"].append(loss.item())
                training_metrics["mlm_loss"].append(mlm_loss.item())
                training_metrics["discriminator_loss"].append(discriminator_loss.item())

                # gradient determination and update
                self.optimizer.zero_grad()

                # determine gradients
                loss.backward()

                if self.optimization_config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model_parameters, max_norm = 1.0)

                # determine gradient norms, equal to one if use_gradient_clipping is set to True
                grads = [p.grad.detach().flatten() for p in self.model_parameters]
                grad_norm = torch.cat(grads).norm()


                # update parameters        
                self.optimizer.step()
                # update learning rate
                self.scheduler.step()

                # determine accuracy metrics, (maybe check for correctness later, has been implemented quickly;))
                with torch.no_grad():
                    # mask to identify ids which have been masked before
                    masked_ids_mask = inputs == self.dataset.tokenizer.mask_token_id
                    predictions = mlm_logits.argmax(-1)
                    mlm_accuracy = (predictions[masked_ids_mask] == generator_labels[masked_ids_mask]).float().mean()
                    active_loss = attention_mask == 1
                    active_logits = discriminator_logits[active_loss]
                    active_predictions = (torch.sign(active_logits) + 1.0) * 0.5
                    active_labels = discriminator_labels[active_loss]
                    discriminator_accuracy = (active_predictions == active_labels).float().mean()
                    discriminator_precision = binary_precision(active_predictions.long(), active_labels.long())
                    discriminator_recall = binary_recall(active_predictions.long(), active_labels.long())


                training_metrics["mlm_accuracy"].append(mlm_accuracy.item())
                training_metrics["discriminator_accuracy"].append(discriminator_accuracy.item())
                training_metrics["discriminator_precision"].append(discriminator_precision.item())
                training_metrics["discriminator_recall"].append(discriminator_recall.item())

                training_metrics["gradient_norm"] = grad_norm.item()
                current_lr = self.scheduler.get_last_lr()[0]
                training_metrics["learning_rates"].append(current_lr)

                if batch_id % 100 == 0:
                    self.logger.info(f"Results after {batch_id/self.iteration_steps_per_epoch:.4%} iterations of epoch {epoch+1}:")
                    self.logger.info(f"Loss: {loss.item():.4f}")
                    self.logger.info(f"MLM Loss: {mlm_loss.item():.4f}")
                    self.logger.info(f"Discriminator Loss: {discriminator_loss.item():.4f}")
                    self.logger.info(f"Gradient norm: {grad_norm:.4f}")
                    self.logger.info(f"Current learning rate: {current_lr}")
                    self.logger.info(f"Accuracy for masking task: {mlm_accuracy.item():.4f}")
                    self.logger.info(f"Accuracy for replacement task: {discriminator_accuracy.item():.4f}")
                    self.logger.info(f"Precision for replacement task: {discriminator_precision.item():.4f}")
                    self.logger.info(f"Recall for replacement task: {discriminator_recall.item():.4f}")
                    self.logger.info("-"*100)   

        self.logger.info("...training is finished, saving results and model.")

        training_metrics_df = pd.DataFrame(training_metrics)
        
        save_path = self._create_directory_and_return_save_path(model_type = "electra")
        # create a function for making an output directory which creates it and saves the csv and model
        training_metrics_df.to_csv(save_path + "training_metrics.csv", index = False)
        training_metrics_df.loc[:, ["loss", "mlm_loss", "discriminator_loss"]].plot(subplots = True)
        plt.savefig(save_path + "loss.png")
        training_metrics_df.loc[:, ["mlm_accuracy", "discriminator_accuracy", "discriminator_precision", "discriminator_recall"]].plot(subplots = True)
        plt.savefig(save_path + "accuracy.png")
        self.generator.save_pretrained(save_path + "mlm_model")
        self.discriminator.save_pretrained(save_path + "discriminator_model")
        self.config.to_json(save_path + "model_config.json")

        self.logger.info("Results and model are saved.")


#########################################################################################################################
# Models for pretraining
#########################################################################################################################


class ElectraAggregationHead(nn.Module):
    """
    Head to aggregate sequence embeddings of a batch to a single prediction. This could be used for every document, e.g., earning call transcripts,
    and the number of sequences is arbitrary per document. Maybe it is better to extent this for a scenario where a batch is actually a batch of batches
    which means multiple documents and their sequences are used for an iteration. Currently, this implementation would create a gradient update per
    document which may cause high variance during training. However, it may also be better to iterate over documents to determine an aggregate loss 
    which is used for a gradient update...discuss layer
    """
    def __init__(self, config):
        super().__init__()
        # densely connect all sequence embeddings
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # dropout
        aggregation_head_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(aggregation_head_dropout)
        self.activation = nn.GELU()
        # project the average aggregate of sequence embeddings after the dense layer
        self.out_projection = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.dropout(x)
        x = self.activation(x)
        x = x.mean(dim = 0)
        x = self.dropout(x)
        x = self.out_projection(x)
        return x
    

class ElectraForAggregatePrediction(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # the encoder for creating token embeddings
        self.electra = ElectraModel(config)
        # the head for creating a single prediction for a batch of embeddings, in our case a batch of sequence embeddings
        self.head = ElectraAggregationHead(config)

        self.post_init()

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Any:
       
        discriminator_hidden_states = self.electra(
            input_ids = input_ids,
            attention_mask=attention_mask,
        )

        # collect all token embeddings 
        sequence_output = discriminator_hidden_states[0]
        # collect the sequence embeddings, only assuming the first token is a seq token
        sequence_output = sequence_output[:, 0, :]
        # logits is the real valued prediction
        logits = self.head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return (loss, logits)


class ElectraSimpleAttention(nn.Module):
    """This class is a single head attention layer"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, sequence_embeddings, return_attention = True):
        query_layer = self.query(sequence_embeddings)
        key_layer = self.key(sequence_embeddings)
        value_layer = self.value(sequence_embeddings)

        # determine attention scores and weights
        attention_scores = torch.matmul(query_layer, key_layer.transpose(0, 1))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) 

        output = (context_layer, attention_probs) if return_attention else (context_layer, )

        return output

class ElectraSimpleAttentionOutput(nn.Module):
    """
    This class is more or less the same as the ElectraSelfOutput, however with the exception, sequence embeddings are aggregated by averaging.
    The layer also applies a residual connection as it is usually done for the combination with attention layers.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # dropout
        aggregation_head_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(aggregation_head_dropout)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out_projection = nn.Linear(config.hidden_size, 1)

    
    def forward(self, sequence_attention_embeddings, sequence_embeddings):
        # sequence attention embeddings are the ones coming from the simple attention layer
        sequence_attention_embeddings = self.dense(sequence_attention_embeddings)
        sequence_attention_embeddings = self.dropout(sequence_attention_embeddings)
        # residual connection with the original sequence embeddings
        sequence_attention_embeddings = self.LayerNorm(sequence_attention_embeddings + sequence_embeddings)
        aggregate_sequence_embeddings = sequence_attention_embeddings.mean(dim = 0)
        logits = self.out_projection(aggregate_sequence_embeddings)
        return logits


class ElectraSimpleAttentionHead(nn.Module):
    """Combination simple attention and output with aggregation"""
    def __init__(self, config):
        super().__init__()
        self.simple_attention = ElectraSimpleAttention(config)
        self.attention_output = ElectraSimpleAttentionOutput(config)

    def forward(self, sequence_embeddings, return_attention = True):

        # determine simple attention output and attention probabilities (if return_attention is set to True)
        simple_attention_output = self.simple_attention(sequence_embeddings, return_attention = return_attention)

        # get sequence attention embeddings after simple attention layer
        sequence_attention_embeddings = simple_attention_output[0]

        if return_attention:
            attention_probs = simple_attention_output[1]

        # process sequence attention embeddings through a dense layer and create a residual connection with the original sequence embeddings
        logits = self.attention_output(sequence_attention_embeddings, sequence_embeddings)
        
        # output prediction and optionally the attention probabilities
        output = (logits, attention_probs) if return_attention else (logits,)
        return output


class ElectraForAggregatePredictionWithAttention(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # the encoder for creating token embeddings
        self.electra = ElectraModel(config)
        # the head for creating a single prediction for a batch of embeddings, in our case a batch of sequence embeddings
        self.head = ElectraSimpleAttentionHead(config)

        self.post_init()

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention = True
    ) -> Any:
       
        discriminator_hidden_states = self.electra(
            input_ids = input_ids,
            attention_mask=attention_mask,
        )

        # collect all token embeddings 
        sequence_output = discriminator_hidden_states[0]
        # collect the sequence embeddings, only assuming the first token is a seq token
        sequence_embeddings = sequence_output[:, 0, :]
        # logits is the real valued prediction
        output = self.head(sequence_embeddings, return_attention = return_attention)
    
        logits = output[0]
    
        if return_attention:
            attention_probabilities = output[1]

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        full_output = (loss, logits, attention_probabilities) if return_attention else (loss, logits)

        return full_output
