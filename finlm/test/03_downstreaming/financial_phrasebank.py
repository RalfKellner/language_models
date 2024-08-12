from datasets import load_dataset
import torch
from finlm.downstreaming import FinetuningEncoderClassifier
from finlm.config import FintuningConfig

if not(torch.cuda.is_available()):
    print("GPU seems to be unavailable.")
else:
    device = torch.device("cuda")

dataset = load_dataset("financial_phrasebank", 'sentences_66agree')
dataset = dataset["train"]

config = FintuningConfig.from_yaml("financial_phrasebank_config.yaml")

model = FinetuningEncoderClassifier(config = config, device = device, dataset = dataset)
model.train_optuna_optimized_cv_model(n_trials=2)