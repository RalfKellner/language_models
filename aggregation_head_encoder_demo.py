from datasets import load_dataset
import re
from finlm.dataset import AggregatedDocumentDataset
from finlm.models import ElectraForAggregatePredictionWithAttention
import torch
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def create_aggregate_dataset(dataset):
    documents = []
    labels = []

    for sample in dataset:
        documents.append(sample["text"])
        labels.append(sample["label"])
        
    documents  = [re.split(r'(?<=[.!?]) +', doc) for doc in documents]

    aggregated_dataset = AggregatedDocumentDataset(
        documents = documents,
        labels = labels,
        tokenizer_path = "google/electra-small-discriminator",
        sequence_length = 128,
        sequence_limit = 128,
        device = "cuda"
    )

    return aggregated_dataset

# Load the dataset
dataset = load_dataset("stanfordnlp/imdb")

# Split the dataset into training and test data
training_dataset = dataset["train"]
test_dataset = dataset["test"]

training_dataset = training_dataset.shuffle(42)
test_dataset = test_dataset.shuffle(42)

logging.info("Preparing aggregated datasets")
training_data = create_aggregate_dataset(training_dataset)
validation_data = create_aggregate_dataset(test_dataset)

device = torch.device("cuda")
n_epochs = 5
warmup_step_fraction = 0.05
learning_rate = 1e-04
use_gradient_clipping = True

logging.info("Initializing model")
model = ElectraForAggregatePredictionWithAttention.from_pretrained("google/electra-small-discriminator", num_labels = 2, classifier_dropout = 0.75)
model.to(device)

n_warmup = int(n_epochs * len(training_data) * warmup_step_fraction)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps = n_epochs * len(training_data), num_warmup_steps = n_warmup)

logging.info("Starting training...")
iteration = 0
for epoch in range(n_epochs):
    training_predictions, training_labels = [], []
    training_loss = 0
    for batch in training_data:
        model_output = model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], labels = batch["labels"].to(torch.long))
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
        training_labels.append(batch["labels"].to(torch.long))
        iteration += 1

    training_loss /= len(training_data)
    logging.info(f"Epoch finished, average loss over training batches: {training_loss:.4f}")
    training_predictions = torch.cat(training_predictions, dim = 0)
    training_labels = torch.cat(training_labels, dim = 0)

    logging.info("-"*100)
    logging.info("Training metrics:")
    logging.info("-"*100)
    if device.type == "cuda":
        training_accuracy = accuracy_score(training_labels.cpu().numpy(), training_predictions.cpu().numpy())
        training_recall = recall_score(training_labels.cpu().numpy(), training_predictions.cpu().numpy())
        training_precision = precision_score(training_labels.cpu().numpy(), training_predictions.cpu().numpy())
        training_f1 = f1_score(training_labels.cpu().numpy(), training_predictions.cpu().numpy())
    else:
        training_accuracy = accuracy_score(training_labels.numpy(), training_predictions.numpy())
        training_recall = recall_score(training_labels.numpy(), training_predictions.numpy())
        training_precision = precision_score(training_labels.numpy(), training_predictions.numpy())
        training_f1 = f1_score(training_labels.numpy(), training_predictions.numpy())
    logging.info(f"Training accuracy: {training_accuracy:.4f}")
    logging.info(f"Training precision: {training_precision:.4f}")
    logging.info(f"Training recall: {training_recall:.4f}")
    logging.info(f"Training f1: {training_f1:.4f}")

    validation_predictions, validation_labels = [], []
    validation_loss = 0
    with torch.no_grad():
        for batch in validation_data:
            model_output = model(input_ids = batch["input_ids"], attention_mask = batch["attention_mask"], labels = batch["labels"].to(torch.long))
            validation_loss += model_output.loss.item()
            batch_predictions = model_output.logits.argmax(dim = 1)
            validation_predictions.append(batch_predictions)
            validation_labels.append(batch["labels"].to(torch.long))
        
    validation_loss /= len(validation_data)
    logging.info(f"Average loss over validation batches: {validation_loss:.4f}")
    validation_predictions = torch.cat(validation_predictions, dim = 0)
    validation_labels = torch.cat(validation_labels, dim = 0)

    logging.info("-"*100)
    logging.info("Validation metrics:")
    logging.info("-"*100)
    if device.type == "cuda":
        validation_accuracy = accuracy_score(validation_labels.cpu().numpy(), validation_predictions.cpu().numpy())
        validation_recall = recall_score(validation_labels.cpu().numpy(), validation_predictions.cpu().numpy())
        validation_precision = precision_score(validation_labels.cpu().numpy(), validation_predictions.cpu().numpy())
        validation_f1 = f1_score(validation_labels.cpu().numpy(), validation_predictions.cpu().numpy())
    else:
        validation_accuracy = accuracy_score(validation_labels.numpy(), validation_predictions.numpy())
        validation_recall = recall_score(validation_labels.numpy(), validation_predictions.numpy())
        validation_precision = precision_score(validation_labels.numpy(), validation_predictions.numpy())
        validation_f1 = f1_score(validation_labels.numpy(), validation_predictions.numpy())
    logging.info(f"Validation accuracy: {validation_accuracy:.4f}")
    logging.info(f"Validation precision: {validation_precision:.4f}")
    logging.info(f"Validation recall: {validation_recall:.4f}")
    logging.info(f"Validation f1: {validation_f1:.4f}")

logging.info("Training ended....")
model.save_pretrained("test_aggregated_model_imdb_2")
logging.info("Model has been saved.")