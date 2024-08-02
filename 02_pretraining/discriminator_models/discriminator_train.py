from finlm.models import PretrainDiscriminator

discriminator_modeling = PretrainDiscriminator("discriminator_config.yaml")
discriminator_modeling.train()