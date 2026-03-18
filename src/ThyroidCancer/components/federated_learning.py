import tensorflow as tf
import numpy as np
import pickle
from src.ThyroidCancer import logger
from src.ThyroidCancer.entity.config_entity import FederatedLearningConfig
from pathlib import Path
from src.ThyroidCancer.utils.model_utils import get_backbone, add_classification_head, get_preprocess_input
from src.ThyroidCancer.utils.losses import categorical_focal_loss

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU is being used ✓")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU found. Training on CPU.")


class FederatedLearning:
    def __init__(self, config: FederatedLearningConfig):
        self.config = config
        self.client_models = []
        self.global_model = None
        
    def _build_model(self):
        """Build backbone model for federated learning"""
        # Load base model
        base_model = get_backbone(
            model_name=self.config.params_model_name,
            input_shape=self.config.params_image_size,
            weights="imagenet",
            include_top=False,
        )
        
        # Freeze base model
        # for layer in base_model.layers:
        #     layer.trainable = False
        for layer in base_model.layers[:-self.config.params_fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[-self.config.params_fine_tune_at:]:
            layer.trainable = True

        model = add_classification_head(base_model, self.config.params_classes)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss=self._get_loss(),
            metrics=["accuracy"]
        )
        
        return model
    
    def partition_data(self):
        """Partition data among clients"""
        import shutil
        import os
        
        logger.info("Partitioning data among clients...")
        
        # Reset client directories to avoid stale class counts
        if Path(self.config.client_data_dir).exists():
            shutil.rmtree(self.config.client_data_dir)

        # Create client directories
        for i in range(self.config.num_clients):
            client_dir = Path(self.config.client_data_dir) / f"client_{i}"
            os.makedirs(client_dir, exist_ok=True)
            
        # Get all class directories
        class_dirs = [d for d in Path(self.config.training_data).iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            images = list(class_dir.glob("*"))
            np.random.shuffle(images)
            
            # Split images among clients
            client_images = np.array_split(images, self.config.num_clients)
            
            for i, imgs in enumerate(client_images):
                client_class_dir = Path(self.config.client_data_dir) / f"client_{i}" / class_name
                os.makedirs(client_class_dir, exist_ok=True)
                
                for img_path in imgs:
                    dest_path = client_class_dir / img_path.name
                    if not dest_path.exists():
                        shutil.copy(img_path, dest_path)

        # Ensure every client has all class directories (even if empty)
        for i in range(self.config.num_clients):
            for class_dir in class_dirs:
                client_class_dir = Path(self.config.client_data_dir) / f"client_{i}" / class_dir.name
                os.makedirs(client_class_dir, exist_ok=True)
                        
        logger.info("Data partitioning completed.")

    def initialize_global_model(self):
        """Initialize the global model"""
        # Partition data first
        self.partition_data()
        
        logger.info("Initializing global model...")
        self.global_model = self._build_model()
        if self.config.base_model_path.exists():
            logger.info(f"Loading pretrained weights from {self.config.base_model_path}...")
            self.global_model.load_weights(str(self.config.base_model_path))
        logger.info("Global model initialized.")
        return self.global_model
    
    def get_client_data_generators(self, client_id):
        """Create data generators for a specific client"""
        client_dir = Path(self.config.client_data_dir) / f"client_{client_id}"
        # Ensure consistent class order across clients
        class_names = sorted([d.name for d in Path(self.config.training_data).iterdir() if d.is_dir()])
        
        preprocess_fn = get_preprocess_input(self.config.params_model_name)
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        train_generator = datagen.flow_from_directory(
            directory=client_dir,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_fl_batch_size,
            class_mode='categorical',
            classes=class_names,
            subset='training',
            shuffle=True
        )
        
        val_generator = datagen.flow_from_directory(
            directory=client_dir,
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_fl_batch_size,
            class_mode='categorical',
            classes=class_names,
            subset='validation',
            shuffle=False
        )
        
        return train_generator, val_generator

    def _get_loss(self):
        loss_name = (self.config.params_loss_name or "").lower()
        if loss_name == "focal":
            return categorical_focal_loss(
                gamma=self.config.params_focal_gamma,
                alpha=self.config.params_focal_alpha,
            )
        return tf.keras.losses.CategoricalCrossentropy()
    
    def train_client(self, client_id, global_weights):
        """Train a client model with global weights"""
        logger.info(f"Training client {client_id}...")
        
        # Create client model and set global weights
        client_model = self._build_model()
        client_model.set_weights(global_weights)
        
        # Get client data
        train_gen, val_gen = self.get_client_data_generators(client_id)

        # Class weights per client
        class_counts = np.bincount(train_gen.classes)
        total = class_counts.sum()
        class_weight = {
            i: total / (len(class_counts) * count) if count > 0 else 1.0
            for i, count in enumerate(class_counts)
        }
        
        # Add learning rate decay callback
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train locally
        history = client_model.fit(
            train_gen,
            epochs=self.config.local_epochs,
            validation_data=val_gen,
            callbacks=[lr_callback],
            class_weight=class_weight,
            verbose=1
        )
        
        # Save client model weights
        client_model_path = Path(self.config.client_models_dir) / f"client_{client_id}_model.h5"
        client_model.save_weights(str(client_model_path))
        logger.info(f"Client {client_id} training complete. Model saved to {client_model_path}")
        
        return client_model.get_weights(), train_gen.samples
    
    def federated_averaging(self, client_weights_list, client_sizes):
        """Aggregate client weights using federated averaging"""
        logger.info("Performing federated averaging...")
        
        avg_weights = []
        total_samples = sum(client_sizes)
        for layer_idx in range(len(client_weights_list[0])):
            layer_weights = [client_weights[layer_idx] for client_weights in client_weights_list]
            weighted = [
                layer_weights[i] * (client_sizes[i] / total_samples)
                for i in range(len(client_sizes))
            ]
            avg_layer_weights = np.sum(weighted, axis=0)
            avg_weights.append(avg_layer_weights)
        
        logger.info("Federated averaging completed.")
        return avg_weights
    
    def run_federated_learning(self):
        """Run the complete federated learning process"""
        logger.info("\n" + "="*80)
        logger.info("STARTING FEDERATED LEARNING")
        logger.info("="*80)
        
        # Initialize global model
        self.initialize_global_model()
        
        # Federated learning rounds
        for round_num in range(1, self.config.fl_rounds + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"FEDERATED LEARNING ROUND {round_num}/{self.config.fl_rounds}")
            logger.info(f"{'='*80}")
            
            global_weights = self.global_model.get_weights()
            client_weights_list = []
            client_sizes = []
            
            # Train each client
            for client_id in range(self.config.num_clients):
                client_weights, client_size = self.train_client(client_id, global_weights)
                client_weights_list.append(client_weights)
                client_sizes.append(client_size)
            
            # Aggregate weights
            aggregated_weights = self.federated_averaging(client_weights_list, client_sizes)
            
            # Update global model
            self.global_model.set_weights(aggregated_weights)
            logger.info(f"Global model updated for round {round_num}.")
        
        # Save final global model weights to avoid serialization issues
        self.global_model.save_weights(self.config.aggregated_model_path)
        logger.info(f"\nFinal global model weights saved to {self.config.aggregated_model_path}")
        logger.info("Federated Learning completed successfully!")
        logger.info("="*80)
