from finlm.models import PretrainElectra

electra_modeling = PretrainElectra("electra_config.yaml")
electra_modeling.train()