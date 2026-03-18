import tensorflow as tf
import numpy as np
import pickle
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from src.ThyroidCancer import logger
from src.ThyroidCancer.entity.config_entity import FeatureSelectionConfig
from pathlib import Path
from src.ThyroidCancer.utils.model_utils import get_backbone, add_classification_head, get_preprocess_input


class FeatureSelection:
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
    
    def _build_full_model(self):
        """Reconstruct the full model architecture matching prepare_base_model"""
        base_model = get_backbone(
            model_name=self.config.params_model_name,
            input_shape=self.config.params_image_size,
            weights="imagenet",
            include_top=False,
        )

        full_model = add_classification_head(base_model, self.config.params_classes)
        
        return full_model
    
    def get_base_model(self):
        """Reconstruct model and load weights"""
        logger.info("Reconstructing model architecture...")
        full_model = self._build_full_model()
        
        logger.info(f"Loading weights from {self.config.base_model_path}...")
        full_model.load_weights(str(self.config.base_model_path))
        logger.info("Model weights loaded successfully.")
        
        return full_model

    def extract_features(self):
        """Extract features from the base model for feature selection"""
        logger.info("Extracting features from base model...")
        
        # Load model
        model = self.get_base_model()
        
        # Create feature extraction model (up to second-to-last layer)
        feature_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.layers[-2].output  # Before final softmax
        )
        
        # Setup data generator
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

        self.generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            shuffle=False,
            class_mode='categorical',
            **dataflow_kwargs
        )
        
        # Extract features
        logger.info("Extracting features...")
        features = feature_model.predict(self.generator, verbose=1)
        labels = self.generator.classes
        
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        
        return features, labels
    
    def select_features(self):
        """Perform feature selection using RFE"""
        logger.info(f"Starting Feature Selection (RFE) to select {self.config.num_features_to_select} features...")
        
        # Extract features
        features, labels = self.extract_features()
        
        # Apply RFE
        estimator = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        selector = RFE(estimator, n_features_to_select=self.config.num_features_to_select, step=50)
        
        selector.fit(features, labels)
        selected_indices = np.where(selector.support_)[0]
        
        logger.info(f"Selected {len(selected_indices)} features.")
        
        # Save selected indices
        with open(self.config.selected_features_path, 'wb') as f:
            pickle.dump(selected_indices, f)
            
        logger.info(f"Selected feature indices saved to {self.config.selected_features_path}")
        return selected_indices
