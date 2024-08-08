from datasets import load_dataset
import torch
from finlm.downstreaming import EncoderClassificationDownstreaming, Hyperparameter

if not(torch.cuda.is_available()):
    print("GPU seems to be unavailable.")
else:
    device = torch.device("cuda")

device = torch.device("cpu")

dataset = load_dataset("financial_phrasebank", 'sentences_66agree')
dataset = dataset["train"]

electra_classifier = EncoderClassificationDownstreaming(
    model_path = "/home/ralf/language_models/pretrained_models/electra_00/discriminator_model",
    num_labels = 3, 
    device = device,
    tokenizer_path = "/home/ralf/language_models/00_tokenizer/finlm-wordpiece-tokenizer.json",
    max_sequence_length=256,
    dataset=dataset,
    text_column="sentence",
    dataset_columns=["input_ids", "attention_mask", "label"],
    shuffle_data=True,
    shuffle_data_random_seed=42, 
    training_data_fraction=0.8,    
    n_epochs = Hyperparameter("n_epochs", int, 2, 10, 5),
    learning_rate = Hyperparameter("learning_rate", float, 1e-05, 1e-04, 5e-05),
    classifier_dropout = Hyperparameter("classifier_dropout", float, 0.2, 0.9, 0.5),
    warmup_step_fraction = Hyperparameter("warmup_step_fraction", float, 0.01, 0.10, 0.05),
    use_gradient_clipping = Hyperparameter("use_gradient_clipping", bool, low = None, high = None, default = False),
    save_path = "/home/ralf/language_models/pretrained_models_downstreaming/financial_phrasebank/electra_00"
)

electra_classifier.train_optuna_optimized_cv_model(n_trials = 25)