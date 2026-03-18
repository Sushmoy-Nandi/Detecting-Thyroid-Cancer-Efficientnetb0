import os
from pathlib import Path
from src.ThyroidCancer.constants import *
from src.ThyroidCancer.utils.common import read_yaml, create_directories
from src.ThyroidCancer.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig,
                                                FeatureSelectionConfig,
                                                FederatedLearningConfig,
                                                ExplainabilityConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        split_dir = Path(config.split_dir)
        train_dir = split_dir / "train"
        test_dir = split_dir / "test"

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            bucket_name=config.bucket_name,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            split_dir=split_dir,
            train_dir=train_dir,
            test_dir=test_dir,
            train_split=self.params.TRAIN_SPLIT,
            test_split=self.params.TEST_SPLIT
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_model_name=self.params.MODEL_NAME,
            params_fine_tune_at=self.params.FINE_TUNE_AT
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = Path(self.config.data_ingestion.split_dir) / "train"
        validation_data = Path(self.config.data_ingestion.split_dir) / "val"
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            validation_data=Path(validation_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_classes=params.CLASSES,
            params_learning_rate=params.LEARNING_RATE,
            params_fine_tune_lr=params.FINE_TUNE_LR,
            params_warmup_epochs=params.WARMUP_EPOCHS,
            params_fine_tune_epochs=params.FINE_TUNE_EPOCHS,
            params_fine_tune_at=params.FINE_TUNE_AT,
            params_model_name=params.MODEL_NAME,
            params_loss_name=params.LOSS_NAME,
            params_focal_gamma=params.FOCAL_GAMMA,
            params_focal_alpha=params.FOCAL_ALPHA
        )

        return training_config

    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/federated_training/aggregated_model.h5",
            training_data=Path(self.config.data_ingestion.split_dir) / "train",
            testing_data=Path(self.config.data_ingestion.split_dir) / "test",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_model_name=self.params.MODEL_NAME,
            params_loss_name=self.params.LOSS_NAME,
            params_focal_gamma=self.params.FOCAL_GAMMA,
            params_focal_alpha=self.params.FOCAL_ALPHA,
            params_tta_steps=self.params.TTA_STEPS
        )
        return eval_config

    def get_feature_selection_config(self) -> FeatureSelectionConfig:
        config = self.config.feature_selection
        prepare_base_model = self.config.prepare_base_model
        training_data = Path(self.config.data_ingestion.split_dir) / "train"
        
        create_directories([config.root_dir])

        feature_selection_config = FeatureSelectionConfig(
            root_dir=Path(config.root_dir),
            selected_features_path=Path(config.selected_features_path),
            base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_classes=self.params.CLASSES,
            params_learning_rate=self.params.LEARNING_RATE,
            num_features_to_select=self.params.NUM_FEATURES_TO_SELECT,
            params_model_name=self.params.MODEL_NAME
        )

        return feature_selection_config

    def get_federated_learning_config(self) -> FederatedLearningConfig:
        federated_learning = self.config.federated_learning
        federated_training = self.config.federated_training
        feature_selection = self.config.feature_selection
        prepare_base_model = self.config.prepare_base_model
        
        create_directories([
            federated_learning.root_dir,
            federated_training.root_dir,
            federated_training.client_models_dir
        ])

        federated_learning_config = FederatedLearningConfig(
            root_dir=Path(federated_learning.root_dir),
            client_data_dir=Path(federated_learning.client_data_dir),
            training_data=Path(federated_learning.training_data),
            aggregated_model_path=Path(federated_training.aggregated_model_path),
            client_models_dir=Path(federated_training.client_models_dir),
            base_model_path=Path(prepare_base_model.updated_base_model_path),
            selected_features_path=Path(feature_selection.selected_features_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
            params_fl_batch_size=self.params.FL_BATCH_SIZE,
            params_classes=self.params.CLASSES,
            params_learning_rate=self.params.LEARNING_RATE,
            num_clients=self.params.NUM_CLIENTS,
            fl_rounds=self.params.FL_ROUNDS,
            local_epochs=self.params.LOCAL_EPOCHS,
            params_model_name=self.params.MODEL_NAME,
            params_loss_name=self.params.LOSS_NAME,
            params_focal_gamma=self.params.FOCAL_GAMMA,
            params_focal_alpha=self.params.FOCAL_ALPHA,
            params_fine_tune_at=self.params.FINE_TUNE_AT
        )

        return federated_learning_config

    def get_explainability_config(self) -> ExplainabilityConfig:
        explainability = self.config.explainability
        federated_training = self.config.federated_training
        training_data = os.path.join(self.config.data_ingestion.split_dir, "test")
        
        create_directories([
            explainability.root_dir,
            explainability.heatmap_dir
        ])

        explainability_config = ExplainabilityConfig(
            root_dir=Path(explainability.root_dir),
            heatmap_dir=Path(explainability.heatmap_dir),
            model_path=Path(federated_training.aggregated_model_path),
            data_dir=Path(training_data),
            params_image_size=self.params.IMAGE_SIZE,
            params_classes=self.params.CLASSES,
            params_model_name=self.params.MODEL_NAME
        )

        return explainability_config
