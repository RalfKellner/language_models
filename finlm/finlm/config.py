from transformers import ElectraConfig
from dataclasses import dataclass, asdict
from typing import Any, Dict
import yaml
import json

# Thanks to ChatGPT:)

@dataclass
class DatasetConfig:
    tokenizer_path: str 
    max_sequence_length: int 
    db_name: str 
    batch_size: int 
    database_retrieval: dict[str, dict[str, int]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetConfig':
        return cls(**data)

@dataclass
class ModelConfig:
    embedding_size: int = 128
    hidden_size: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    generator_size: float = 0.25
    generator_layer_size: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return cls(**data)

@dataclass
class OptimizationConfig:
    learning_rate: float = 0.0001
    n_epochs: int = 1
    lr_scheduler_warm_up_steps: int = 1000
    mlm_probability: float = 0.15
    use_gradient_clipping: bool = True
    discriminator_weight: int = 50
    discriminator_sampling: str = "multinomial"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        return cls(**data)

@dataclass
class FinLMConfig:
    dataset_config: DatasetConfig
    model_config: ModelConfig
    optimization_config: OptimizationConfig
    save_models_and_results_to: str

    @classmethod
    def from_yaml(cls, config_file_path: str, save_root_path: str) -> 'FinLMConfig':
        with open(config_file_path, 'r') as file:
            config_data = yaml.safe_load(file)

        dataset_config = DatasetConfig.from_dict(config_data.get('dataset_config', {}))
        model_config = ModelConfig.from_dict(config_data.get('model_config', {}))
        optimization_config = OptimizationConfig.from_dict(config_data.get('optimization_config', {}))

        return cls(
            dataset_config=dataset_config,
            model_config=model_config,
            optimization_config=optimization_config,
            save_models_and_results_to = save_root_path
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_config': asdict(self.dataset_config),
            'model_config': asdict(self.model_config),
            'optimization_config': asdict(self.optimization_config)
        }

    def to_json(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)


