from finlm.models import PretrainMLM
from finlm.config import FinLMConfig
from finlm.callbacks import CallbackManager, MlMWandBTrackerCallback, CallbackTypes

config = FinLMConfig.from_yaml("mlm_config.yaml", "/data/language_models/pretrained_models/")

mlm_modeling = PretrainMLM(config)

cb = MlMWandBTrackerCallback(CallbackTypes.ON_BATCH_END, project_name="mlm-pretraining", name="initial-run",
                             api_key="7553704685a7f606bb03f1b95095a619a23cf5cd", entity="finlm",
                             batch_size=config.dataset_config.batch_size)

mlm_modeling.add_callback(cb)
mlm_modeling.train()