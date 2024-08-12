from finlm.models import PretrainDiscriminator
from finlm.config import FinLMConfig

config = FinLMConfig.from_yaml("discriminator.yaml", "/data/language_models/pretrained_models/")

discriminator_modeling = PretrainDiscriminator(config)
discriminator_modeling.train()