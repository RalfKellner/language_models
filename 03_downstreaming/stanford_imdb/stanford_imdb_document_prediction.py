from datasets import load_dataset
import torch
from finlm.models import ElectraDocumentClassification
from finlm.dataset import FinetuningDocumentDataset
from finlm.downstreaming import FinetuningEncoderClassifier
from finlm.config import FintuningConfig
import re

model_loader = lambda model_path, num_labels, classifier_dropout: ElectraDocumentClassification.from_pretrained(model_path, num_labels = num_labels, classifier_dropout = classifier_dropout, num_sequence_attention_heads = 1) 

if not(torch.cuda.is_available()):
    print("GPU seems to be unavailable.")
else:
    device = torch.device("cuda")

from datasets import load_dataset

# Load the dataset
dataset = load_dataset("stanfordnlp/imdb")

# Split the dataset into training and test data
training_data = dataset["train"]
test_data = dataset["test"]

# datasets must be shuffled, because they are sorted by label
training_data = training_data.shuffle(42)
test_data = test_data.shuffle(42)


training_documents, training_labels = [], []
for sample in training_data:
    training_documents.append(sample["text"])
    training_labels.append(sample["label"])

test_documents, test_labels = [], []
for sample in test_data:
    test_documents.append(sample["text"])
    test_labels.append(sample["label"])

training_documents = [re.split(r'(?<=[.!?]) +', doc) for doc in training_documents]
test_documents = [re.split(r'(?<=[.!?]) +', doc) for doc in test_documents]

documents = training_documents + test_documents
labels = training_labels + test_labels

config = FintuningConfig.from_yaml("stanford_imdb.yaml")
dataset = FinetuningDocumentDataset(documents = documents, labels =  labels, tokenizer_path = config.tokenizer_path, sequence_length = config.max_sequence_length)

model = FinetuningEncoderClassifier(config = config, device = device, dataset = dataset, model_loader = model_loader)
model.train_optuna_optimized_cv_model(n_trials=10)