from finlm.models import PretrainMLM
from finlm.config import FinLMConfig

config = FinLMConfig.from_yaml("mlm_config.yaml", "/data/language_models/test_outputs/02_pretraining_results/")

mlm_modeling = PretrainMLM(config)
mlm_modeling.train()