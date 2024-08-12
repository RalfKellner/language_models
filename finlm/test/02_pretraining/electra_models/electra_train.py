from finlm.models import PretrainElectra
from finlm.config import FinLMConfig

config = FinLMConfig.from_yaml("electra_config.yaml", "/data/language_models/test_outputs/02_pretraining_results/")

electra_modeling = PretrainElectra(config)
electra_modeling.train()