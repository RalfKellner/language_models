import os
import yaml
from finlm.dataset import FinLMDataset
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining
from transformers import get_linear_schedule_with_warmup
from finlm.utils import mask_tokens, replace_masked_tokens_randomly, replace_masked_tokens_from_generator
import pandas as pd
import numpy as np
import torch
from torcheval.metrics.functional import binary_precision, binary_recall
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#########################################################################################################################
# Models for pretraining
#########################################################################################################################


class PretrainLM:
    def __init__(self, config_path):
        self.config_path = config_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self._set_device()
        self._load_config()

    def load_dataset(self, **kwargs):
        self.dataset = FinLMDataset(
            tokenizer_path = self.config["dataset_config"]["tokenizer_path"],
            max_sequence_length = self.config["dataset_config"]["max_sequence_length"],
            db_name = self.config["dataset_config"]["db_name"],
            n_10k_seq = self.config["dataset_config"]["n_10k_seq"],
            n_8k_seq = self.config["dataset_config"]["n_8k_seq"],
            n_ec_seq = self.config["dataset_config"]["n_ec_seq"],
            n_news_seq = self.config["dataset_config"]["n_news_seq"],
            batch_size = self.config["dataset_config"]["batch_size"],
            random_sql = self.config["dataset_config"]["draw_random_sequences"],
            **kwargs
        )

    def _set_device(self):
        if not(torch.cuda.is_available()):
            logging.warning("GPU seems to be unavailable.")
        else:
            self.device = torch.device("cuda")

    def _load_config(self):
        with open(self.config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        self.config = config

    def _create_directory_and_return_save_path(self, model_type):
        n_sequences = self.config["dataset_config"]["n_10k_seq"] + self.config["dataset_config"]["n_8k_seq"] + self.config["dataset_config"]["n_ec_seq"] + self.config["dataset_config"]["n_news_seq"]
        config_name_part = f'max_seq_len_{self.config["dataset_config"]["max_sequence_length"]}_n_seq_{n_sequences}_batch_size_{self.config["dataset_config"]["batch_size"]}'
        model_name_part = f'_emb_{self.config["model_config"]["embedding_size"]}_hidden_{self.config["model_config"]["hidden_size"]}'
        optim_name_part = f'_lr_{self.config["optimization_config"]["learning_rate"]}_epochs_{self.config["optimization_config"]["n_epochs"]}'
        full_name = f"{model_type}_" + config_name_part + model_name_part + optim_name_part

        save_path = f"../../pretrained_models/{full_name}"
        os.mkdir(save_path)
        return save_path


class PretrainMLM(PretrainLM):
    def __init__(self, config_path):
        super().__init__(config_path)
        self._load_config()
        self.prepare_data_model_optimizer()

    def load_model(self):
        self.model_config = ElectraConfig(
            vocab_size = self.dataset.tokenizer.vocab_size,
            embedding_size = self.config["model_config"]["embedding_size"],
            hidden_size = self.config["model_config"]["hidden_size"], 
            num_hidden_layers = self.config["model_config"]["num_hidden_layers"],
            num_attention_heads = self.config["model_config"]["num_attention_heads"]
        )

        self.model = ElectraForMaskedLM(self.model_config)
        self.model.to(self.device)

    def load_optimization(self):
        self.iteration_steps_per_epoch = int(np.ceil(sum(self.dataset.n_sequences) / self.dataset.batch_size))
        self.n_epochs = self.config["optimization_config"]["n_epochs"]  
        total_steps = self.iteration_steps_per_epoch * self.n_epochs  
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.config["optimization_config"]["learning_rate"]) 
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = self.config["optimization_config"]["lr_scheduler_warm_up_steps"], num_training_steps=total_steps)

    def prepare_data_model_optimizer(self):
        self.load_dataset()
        self.load_model()
        self.load_optimization()

    def train(self):
        random_sql = self.config["dataset_config"]["draw_random_sequences"]
        if not random_sql:
            new_database_at_every_epoch = True
        else:
            new_database_at_every_epoch = self.config["optimization_config"]["new_database_for_every_epoch"]
        mlm_probability = self.config["optimization_config"]["mlm_probability"]

        special_token_ids = set(self.dataset.tokenizer.all_special_ids).difference(set([self.dataset.tokenizer.mask_token_id]))
        use_gradient_clipping = self.config["optimization_config"]["use_gradient_clipping"]

        self.logger.info("Starting with training...")
        loss, accuracy, gradient_norms, learning_rates = [], [], [], []
        for epoch in range(self.n_epochs): 
            if (epoch  > 0):
                if not random_sql:
                    self.load_dataset(offsets = [(seq * epoch) for seq in self.dataset.n_sequences])
                elif new_database_at_every_epoch:
                    self.load_dataset()

            for batch_id, batch in enumerate(self.dataset):
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                inputs, labels = mask_tokens(
                    inputs,
                    mlm_probability = mlm_probability,
                    mask_token_id = self.dataset.tokenizer.mask_token_id,
                    special_token_ids = special_token_ids,
                    n_tokens = self.dataset.tokenizer.vocab_size)

                mlm_output = self.model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                mlm_loss, mlm_logits = mlm_output.loss, mlm_output.logits
                loss.append(mlm_loss.item())

                # gradient determination and update
                self.optimizer.zero_grad()

                # determine gradients
                mlm_loss.backward()

                if use_gradient_clipping:
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

                accuracy.append(mlm_accuracy.item())
                gradient_norms.append(mlm_grad_norm.item())
                current_lr = self.scheduler.get_last_lr()[0]
                learning_rates.append(current_lr)

                if batch_id % 100 == 0:
                    self.logger.info(f"Results after {batch_id/self.iteration_steps_per_epoch:.4%} iterations of epoch {epoch+1}:")
                    self.logger.info(f"MLM loss: {mlm_loss.item():.4f}")
                    self.logger.info(f"Gradient norm: {mlm_grad_norm:.4f}")
                    self.logger.info(f"Current learning rate: {current_lr}")
                    self.logger.info(f"Accuracy for masking task: {mlm_accuracy.item():.4f}")
                    self.logger.info("-"*100)   

        self.logger.info("...training is finished, saving results and model.")

        training_metrics_df = pd.DataFrame({
            "loss": loss,
            "accuracy": accuracy,
            "gradient_norm": gradient_norms,
            "learning_rate": learning_rates
            })
        
        save_path = self._create_directory_and_return_save_path(model_type = "mlm")
        # create a function for making an output directory which creates it and saves the csv and model
        training_metrics_df.to_csv(save_path + "/training_metrics.csv", index = False)
        self.model.save_pretrained(save_path + "/mlm_model")

        self.logger.info("Results and model are saved.")



class PretrainDiscriminator(PretrainLM):
    def __init__(self, config_path):
        super().__init__(config_path)
        self._load_config()
        self.prepare_data_model_optimizer()

    def load_model(self):
        self.model_config = ElectraConfig(
            vocab_size = self.dataset.tokenizer.vocab_size,
            embedding_size = self.config["model_config"]["embedding_size"],
            hidden_size = self.config["model_config"]["hidden_size"], 
            num_hidden_layers = self.config["model_config"]["num_hidden_layers"],
            num_attention_heads = self.config["model_config"]["num_attention_heads"]
        )

        self.model = ElectraForPreTraining(self.model_config)
        self.model.to(self.device)

    def load_optimization(self):
        self.iteration_steps_per_epoch = int(np.ceil(sum(self.dataset.n_sequences) / self.dataset.batch_size))
        self.n_epochs = self.config["optimization_config"]["n_epochs"]  
        total_steps = self.iteration_steps_per_epoch * self.n_epochs  
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.config["optimization_config"]["learning_rate"]) 
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = self.config["optimization_config"]["lr_scheduler_warm_up_steps"], num_training_steps=total_steps)

    def prepare_data_model_optimizer(self):
        self.load_dataset()
        self.load_model()
        self.load_optimization()

    def train(self):
        random_sql = self.config["dataset_config"]["draw_random_sequences"]
        if not random_sql:
            new_database_at_every_epoch = True
        else:
            new_database_at_every_epoch = self.config["optimization_config"]["new_database_for_every_epoch"]
        mlm_probability = self.config["optimization_config"]["mlm_probability"]

        special_token_ids = set(self.dataset.tokenizer.all_special_ids).difference(set([self.dataset.tokenizer.mask_token_id]))
        use_gradient_clipping = self.config["optimization_config"]["use_gradient_clipping"]

        self.logger.info("Starting with training...")
        loss, accuracy, precision, recall, gradient_norms, learning_rates = [], [], [], [], [], []
        for epoch in range(self.n_epochs): 
            if (epoch  > 0):
                if not random_sql:
                    self.load_dataset(offsets = [(seq * epoch) for seq in self.dataset.n_sequences])
                elif new_database_at_every_epoch:
                    self.load_dataset()

            for batch_id, batch in enumerate(self.dataset):
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                inputs, labels = replace_masked_tokens_randomly(
                    inputs, 
                    mlm_probability = mlm_probability,
                    mask_token_id = self.dataset.tokenizer.mask_token_id,
                    special_token_ids = special_token_ids,
                    n_tokens = self.dataset.tokenizer.vocab_size,
                    hard_masking = True
                )

                discriminator_output = self.model(input_ids = inputs, attention_mask = attention_mask, labels = labels)
                discriminator_loss, discriminator_logits = discriminator_output.loss, discriminator_output.logits
                loss.append(discriminator_loss.item())

                # gradient determination and update
                self.optimizer.zero_grad()

                # determine gradients
                discriminator_loss.backward()

                if use_gradient_clipping:
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

                accuracy.append(discriminator_accuracy.item())
                precision.append(discriminator_precision.item())
                recall.append(discriminator_recall.item())
                gradient_norms.append(discriminator_grad_norm.item())
                current_lr = self.scheduler.get_last_lr()[0]
                learning_rates.append(current_lr)

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

        training_metrics_df = pd.DataFrame({
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "gradient_norm": gradient_norms,
            "learning_rate": learning_rates
            })
        
        save_path = self._create_directory_and_return_save_path(model_type = "discriminator")
        training_metrics_df.to_csv(save_path + "/training_metrics.csv", index = False)
        self.model.save_pretrained(save_path + "/discriminator_model")

        self.logger.info("Results and model are saved.")
        

class PretrainElectra(PretrainLM):
    def __init__(self, config_path):
        super().__init__(config_path)
        self._load_config()
        self.prepare_data_model_optimizer()

    def load_model(self):
        self.generator_model_config = ElectraConfig(
            vocab_size = self.dataset.tokenizer.vocab_size,
            embedding_size = self.config["model_config"]["embedding_size"],
            hidden_size = int(self.config["model_config"]["hidden_size"] * self.config["model_config"]["generator_size"]), 
            intermediate_size = int(self.config["model_config"]["intermediate_size"] * self.config["model_config"]["generator_size"]),
            num_hidden_layers = int(self.config["model_config"]["num_hidden_layers"] * self.config["model_config"]["generator_layer_size"]),
            num_attention_heads = int(self.config["model_config"]["num_attention_heads"] * self.config["model_config"]["generator_size"])
        )

        self.discriminator_model_config = ElectraConfig(
            vocab_size = self.dataset.tokenizer.vocab_size,
            embedding_size = self.config["model_config"]["embedding_size"],
            hidden_size = self.config["model_config"]["hidden_size"], 
            num_hidden_layers = self.config["model_config"]["num_hidden_layers"],
            num_attention_heads = self.config["model_config"]["num_attention_heads"]
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
                
        self.iteration_steps_per_epoch = int(np.ceil(sum(self.dataset.n_sequences) / self.dataset.batch_size))
        self.n_epochs = self.config["optimization_config"]["n_epochs"]  
        total_steps = self.iteration_steps_per_epoch * self.n_epochs  
        self.optimizer = torch.optim.AdamW(self.model_parameters, lr = self.config["optimization_config"]["learning_rate"]) 
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = self.config["optimization_config"]["lr_scheduler_warm_up_steps"], num_training_steps=total_steps)

    def prepare_data_model_optimizer(self):
        self.load_dataset()
        self.load_model()
        self.load_optimization()

    def train(self):
        random_sql = self.config["dataset_config"]["draw_random_sequences"]
        if not random_sql:
            new_database_at_every_epoch = True
        else:
            new_database_at_every_epoch = self.config["optimization_config"]["new_database_for_every_epoch"]
        mlm_probability = self.config["optimization_config"]["mlm_probability"]
        discriminator_weight = self.config["optimization_config"]["discriminator_weight"]
        discriminator_sampling = self.config["optimization_config"]["discriminator_sampling"]

        special_token_ids = set(self.dataset.tokenizer.all_special_ids).difference(set([self.dataset.tokenizer.mask_token_id]))
        use_gradient_clipping = self.config["optimization_config"]["use_gradient_clipping"]

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

        for epoch in range(self.n_epochs): 
            if (epoch  > 0):
                if not random_sql:
                    self.load_dataset(offsets = [(seq * epoch) for seq in self.dataset.n_sequences])
                elif new_database_at_every_epoch:
                    self.load_dataset()

            for batch_id, batch in enumerate(self.dataset):
                inputs, attention_mask = batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

                original_inputs = inputs.clone()
                generator_inputs, generator_labels = mask_tokens(
                    inputs,
                    mlm_probability = mlm_probability,
                    mask_token_id = self.dataset.tokenizer.mask_token_id,
                    special_token_ids = special_token_ids,
                    n_tokens = self.dataset.tokenizer.vocab_size)

                mlm_output = self.generator(input_ids = generator_inputs, attention_mask = attention_mask, labels = generator_labels)
                mlm_loss, mlm_logits = mlm_output.loss, mlm_output.logits

                sampling_logits = mlm_logits.detach()
                discriminator_inputs, discriminator_labels = replace_masked_tokens_from_generator(
                    masked_inputs = generator_inputs,
                    original_inputs = original_inputs,
                    logits = sampling_logits,
                    special_mask_id = self.dataset.tokenizer.mask_token_id,
                    discriminator_sampling = discriminator_sampling
                    )
                
                discriminator_output = self.discriminator(input_ids = discriminator_inputs, attention_mask = attention_mask, labels = discriminator_labels)
                discriminator_loss, discriminator_logits = discriminator_output.loss, discriminator_output.logits

                loss = mlm_loss + discriminator_weight * discriminator_loss

                training_metrics["loss"].append(loss.item())
                training_metrics["mlm_loss"].append(mlm_loss.item())
                training_metrics["discriminator_loss"].append(discriminator_loss.item())

                # gradient determination and update
                self.optimizer.zero_grad()

                # determine gradients
                loss.backward()

                if use_gradient_clipping:
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
        training_metrics_df.to_csv(save_path + "/training_metrics.csv", index = False)
        self.generator.save_pretrained(save_path + "/mlm_model")
        self.discriminator.save_pretrained(save_path + "/discriminator_model")

        self.logger.info("Results and model are saved.")
