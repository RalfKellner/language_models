from finlm.models import PreTrainGpt
from finlm.config import FinLMGptConfig

config = FinLMGptConfig.from_yaml("gpt_config.yaml", "/data/language_models/pretrained_models/")

gpt_modeling = PreTrainGpt(config)
gpt_modeling.train()