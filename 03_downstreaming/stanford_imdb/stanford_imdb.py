from datasets import load_dataset, concatenate_datasets
import torch
from finlm.downstreaming import FinetuningEncoderClassifier
from finlm.config import FintuningConfig
from finlm.dataset import FinetuningDataset
from transformers import ElectraForSequenceClassification

model_loader = lambda model_path, num_labels, classifier_dropout: ElectraForSequenceClassification.from_pretrained(model_path, num_labels = num_labels, classifier_dropout = classifier_dropout) 

if not(torch.cuda.is_available()):
    print("GPU seems to be unavailable.")
else:
    device = torch.device("cuda")

# Load the dataset
dataset = load_dataset("stanfordnlp/imdb")

# Split the dataset into training and test data
training_data = dataset["train"]
test_data = dataset["test"]

# datasets must be shuffled, because they are sorted by label
training_data = training_data.shuffle(42)
test_data = test_data.shuffle(42)

dataset = concatenate_datasets([training_data, test_data])

config = FintuningConfig.from_yaml("stanford_imdb.yaml")
dataset = FinetuningDataset(
    tokenizer_path = config.tokenizer_path,
    max_sequence_length = config.max_sequence_length,
    dataset = dataset,
    text_column = config.text_column,
    dataset_columns = config.dataset_columns
)

model = FinetuningEncoderClassifier(config = config, device = device, dataset = dataset, model_loader = model_loader)
model.train_optuna_optimized_cv_model(n_trials=10)