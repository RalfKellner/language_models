from finlm.models import PretrainDiscriminator
from finlm.config import FinLMConfig

config = FinLMConfig.from_yaml("discriminator_config.yaml", "../../test_pretrained_models/")

discriminator_modeling = PretrainDiscriminator(config)
discriminator_modeling.train()