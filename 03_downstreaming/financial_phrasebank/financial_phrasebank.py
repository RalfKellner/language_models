from datasets import load_dataset
import torch
from finlm.downstreaming import FinetuningEncoderClassifier
from finlm.config import FintuningConfig
from transformers import RobertaForSequenceClassification

model_loader = lambda model_path, num_labels, classifier_dropout: RobertaForSequenceClassification.from_pretrained(model_path, num_labels = num_labels, classifier_dropout = classifier_dropout) 

if not(torch.cuda.is_available()):
    print("GPU seems to be unavailable.")
else:
    device = torch.device("cuda")

dataset = load_dataset("financial_phrasebank", 'sentences_66agree')
dataset = dataset["train"]

config = FintuningConfig.from_yaml("financial_phrasebank_config.yaml")

model = FinetuningEncoderClassifier(config = config, device = device, dataset = dataset, model_loader = model_loader)
model.train_optuna_optimized_cv_model(n_trials=25)