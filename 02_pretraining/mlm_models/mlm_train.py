from finlm.models import PretrainMLM

mlm_modeling = PretrainMLM("mlm_config.yaml")
mlm_modeling.train()