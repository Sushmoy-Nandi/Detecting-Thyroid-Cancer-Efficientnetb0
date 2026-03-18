from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    bucket_name: str
    local_data_file: Path
    unzip_dir: Path
    split_dir: Path
    train_dir: Path
    test_dir: Path
    train_split: float
    test_split: float

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_model_name: str
    params_fine_tune_at: int

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    validation_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    params_classes: int
    params_learning_rate: float
    params_fine_tune_lr: float
    params_warmup_epochs: int
    params_fine_tune_epochs: int
    params_fine_tune_at: int
    params_model_name: str
    params_loss_name: str
    params_focal_gamma: float
    params_focal_alpha: float

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    testing_data: Path
    all_params: dict
    params_image_size: list
    params_batch_size: int
    params_model_name: str
    params_loss_name: str
    params_focal_gamma: float
    params_focal_alpha: float
    params_tta_steps: int

@dataclass(frozen=True)
class FeatureSelectionConfig:
    root_dir: Path
    selected_features_path: Path
    base_model_path: Path
    training_data: Path
    params_image_size: list
    params_batch_size: int
    params_classes: int
    params_learning_rate: float
    num_features_to_select: int
    params_model_name: str

@dataclass(frozen=True)
class FederatedLearningConfig:
    root_dir: Path
    client_data_dir: Path
    training_data: Path
    aggregated_model_path: Path
    client_models_dir: Path
    base_model_path: Path
    selected_features_path: Path
    params_image_size: list
    params_batch_size: int
    params_fl_batch_size: int
    params_classes: int
    params_learning_rate: float
    num_clients: int
    fl_rounds: int
    local_epochs: int
    params_model_name: str
    params_loss_name: str
    params_focal_gamma: float
    params_focal_alpha: float
    params_fine_tune_at: int

@dataclass(frozen=True)
class ExplainabilityConfig:
    root_dir: Path
    heatmap_dir: Path
    model_path: Path
    data_dir: Path
    params_image_size: list
    params_classes: int
    params_model_name: str
