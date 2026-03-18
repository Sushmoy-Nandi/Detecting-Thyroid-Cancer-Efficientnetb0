import tensorflow as tf
import numpy as np
from pathlib import Path
from src.ThyroidCancer.entity.config_entity import EvaluationConfig
from src.ThyroidCancer.utils.common import save_json
from src.ThyroidCancer.utils.model_utils import get_backbone, add_classification_head, get_preprocess_input
from src.ThyroidCancer.utils.losses import categorical_focal_loss



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

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

        eval_dir = self.config.testing_data or self.config.training_data
        self.eval_dir = eval_dir
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=eval_dir,
            class_mode="categorical",
            shuffle=False,
            **dataflow_kwargs
        )

    
    def _build_model(self) -> tf.keras.Model:
        base_model = get_backbone(
            model_name=self.config.params_model_name,
            input_shape=self.config.params_image_size,
            weights="imagenet",
            include_top=False,
        )

        model = add_classification_head(base_model, self.config.all_params.CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=self._get_loss(),
            metrics=["accuracy"]
        )
        return model
    

    def evaluation(self):
        model = self._build_model()
        model.load_weights(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)
        self.extra_metrics = self._compute_extra_metrics(model)

    
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        if getattr(self, "extra_metrics", None):
            scores.update(self.extra_metrics)
        save_json(path=Path("scores.json"), data=scores)

    def _predict_with_tta(self, model, generator):
        tta_steps = max(int(self.config.params_tta_steps or 1), 1)
        if tta_steps == 1:
            return model.predict(generator)

        preprocess_fn = get_preprocess_input(self.config.params_model_name)
        tta_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
        )

        tta_generator = tta_datagen.flow_from_directory(
            directory=self.eval_dir,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical",
            shuffle=False,
        )

        preds = []
        for _ in range(tta_steps):
            tta_generator.reset()
            preds.append(model.predict(tta_generator))
        return np.mean(preds, axis=0)

    def _compute_extra_metrics(self, model):
        y_prob = self._predict_with_tta(model, self.valid_generator)
        y_true = self.valid_generator.classes
        y_pred = np.argmax(y_prob, axis=1)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        auc_metric = tf.keras.metrics.AUC()
        if y_prob.shape[1] >= 2:
            auc_metric.update_state(y_true, y_prob[:, 1])
            auc = float(auc_metric.result().numpy())
        else:
            auc = 0.0

        return {
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "auc": float(auc),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tta_steps": int(self.config.params_tta_steps or 1),
        }

    def _get_loss(self):
        loss_name = (self.config.params_loss_name or "").lower()
        if loss_name == "focal":
            return categorical_focal_loss(
                gamma=self.config.params_focal_gamma,
                alpha=self.config.params_focal_alpha,
            )
        return tf.keras.losses.CategoricalCrossentropy()

    

    
