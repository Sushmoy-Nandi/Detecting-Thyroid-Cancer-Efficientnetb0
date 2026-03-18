import os
import tensorflow as tf
from src.ThyroidCancer.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path
from src.ThyroidCancer.utils.model_utils import get_backbone, add_classification_head


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Save full model without optimizer state to avoid serialization issues
        model.save(path, include_optimizer=False)

    @staticmethod
    def save_weights(path: Path, model: tf.keras.Model):
        model.save_weights(path)


    
    def get_base_model(self):
        self.model = get_backbone(
            model_name=self.config.params_model_name,
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
        )

        # Save base model as weights only (lighter, avoids config serialization issues)
        self.save_weights(path=self.config.base_model_path, model=self.model)

    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # Freeze base model layers
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        full_model = add_classification_head(model, classes)

        # Compile with Adam optimizer (better than SGD for most cases)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=self.config.params_fine_tune_at,
            learning_rate=self.config.params_learning_rate
        )

        # Save updated model as weights to avoid serialization issues
        self.save_weights(path=self.config.updated_base_model_path, model=self.full_model)
