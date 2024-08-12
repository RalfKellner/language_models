from finlm.models import PretrainDiscriminator
from finlm.config import FinLMConfig

config = FinLMConfig.from_yaml("discriminator_config.yaml", "/data/language_models/test_outputs/02_pretraining_results/")

discriminator_modeling = PretrainDiscriminator(config)
discriminator_modeling.train()