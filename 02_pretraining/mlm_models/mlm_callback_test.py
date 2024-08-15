from finlm.models import PretrainMLM
from finlm.config import FinLMConfig
from finlm.callbacks import CallbackManager

config = FinLMConfig.from_yaml("mlm_config.yaml", "/data/language_models/pretrained_models/")

mlm_modeling = PretrainMLM(config)
mlm_modeling.train()