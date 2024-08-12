from finlm.models import PretrainMLM
from finlm.config import FinLMConfig

config = FinLMConfig.from_yaml("mlm_config.yaml", "../../pretrained_models/")

mlm_modeling = PretrainMLM(config)
mlm_modeling.train()