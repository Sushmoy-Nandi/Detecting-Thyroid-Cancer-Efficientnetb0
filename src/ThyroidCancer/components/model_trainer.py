import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from src.ThyroidCancer.entity.config_entity import TrainingConfig
from pathlib import Path
import numpy as np
from src.ThyroidCancer.utils.model_utils import get_backbone, add_classification_head, get_preprocess_input
from src.ThyroidCancer.utils.losses import categorical_focal_loss


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        base_model = get_backbone(
            model_name=self.config.params_model_name,
            input_shape=self.config.params_image_size,
            weights="imagenet",
            include_top=False,
        )

        self.model = add_classification_head(base_model, self.config.params_classes)

        self.model.load_weights(self.config.updated_base_model_path)

    def train_valid_generator(self):

        preprocess_fn = get_preprocess_input(self.config.params_model_name)
        datagenerator_kwargs = dict(preprocessing_function=preprocess_fn)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.validation_data,
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                brightness_range=(0.8, 1.2),
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            shuffle=True,
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Class weights to handle imbalance
        classes = self.train_generator.classes
        class_counts = np.bincount(classes)
        total = class_counts.sum()
        class_weight = {
            i: total / (len(class_counts) * count) if count > 0 else 1.0
            for i, count in enumerate(class_counts)
        }

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            )
        ]

        # Stage 1: warmup with frozen base
        for layer in self.model.layers:
            layer.trainable = False
        metrics = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=self._get_loss(),
            metrics=metrics
        )
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_warmup_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            class_weight=class_weight,
            callbacks=callbacks
        )

        # Stage 2: fine-tune last N layers
        for layer in self.model.layers[:-self.config.params_fine_tune_at]:
            layer.trainable = False
        for layer in self.model.layers[-self.config.params_fine_tune_at:]:
            layer.trainable = True
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_fine_tune_lr),
            loss=self._get_loss(),
            metrics=metrics
        )
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_fine_tune_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            class_weight=class_weight,
            callbacks=callbacks
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    def _get_loss(self):
        loss_name = (self.config.params_loss_name or "").lower()
        if loss_name == "focal":
            return categorical_focal_loss(
                gamma=self.config.params_focal_gamma,
                alpha=self.config.params_focal_alpha,
            )
        return tf.keras.losses.CategoricalCrossentropy()
