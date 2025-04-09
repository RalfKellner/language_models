from finlm.models import PreTrainGpt
from finlm.config import FinLMGptConfig

config = FinLMGptConfig.from_yaml("gpt_config.yaml", "/data/language_models/test_outputs/02_pretraining_results/")

gpt_modeling = PreTrainGpt(config)
gpt_modeling.train()