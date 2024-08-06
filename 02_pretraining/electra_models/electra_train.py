from finlm.models import PretrainElectra
from finlm.config import FinLMConfig

config = FinLMConfig.from_yaml("electra_config.yaml", "../../pretrained_models/")

electra_modeling = PretrainElectra(config)
electra_modeling.train()