import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.ThyroidCancer import logger
from src.ThyroidCancer.entity.config_entity import ExplainabilityConfig
import cv2
import glob
import random
from pathlib import Path
from src.ThyroidCancer.utils.model_utils import (
    get_backbone,
    add_classification_head,
    get_preprocess_input,
    get_last_conv_layer_name,
)


class Explainability:
    def __init__(self, config: ExplainabilityConfig):
        self.config = config
        self.model = None
        
    def _build_model(self):
        """Build the model architecture matching the trained model"""
        base_model = get_backbone(
            model_name=self.config.params_model_name,
            input_shape=self.config.params_image_size,
            weights="imagenet",
            include_top=False,
        )

        model = add_classification_head(base_model, self.config.params_classes)
        
        return model

    def load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from {self.config.model_path}...")
        self.model = self._build_model()
        self.model.load_weights(str(self.config.model_path))
        logger.info("Model loaded successfully.")
        
    def make_gradcam_heatmap(self, img_array, last_conv_layer_name="top_activation", pred_index=None):
        """Generate Grad-CAM heatmap"""
        # Create a model that maps the input image to the activations of the last conv layer
        # as well as the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )

        # Compute gradient of top predicted class for our input image
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # Gradient of the output neuron with regard to the output feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # Vector of mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Multiply each channel by importance and sum all channels
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def save_and_display_gradcam(self, img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
        """Save and display Grad-CAM visualization"""
        # Load the original image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.config.params_image_size[0], self.config.params_image_size[1]))
        
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)
        logger.info(f"Saved Grad-CAM image to {cam_path}")
        
        # Display
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title("Grad-CAM")
        plt.show()

    def generate_explanations(self, num_samples=5):
        """Generate Grad-CAM explanations for random sample images"""
        logger.info(f"Generating Grad-CAM explanations for {num_samples} samples...")
        
        # Get random images from data directory
        all_images = glob.glob(str(Path(self.config.data_dir) / "**" / "*.jpg"), recursive=True)
        all_images.extend(glob.glob(str(Path(self.config.data_dir) / "**" / "*.png"), recursive=True))
        
        if len(all_images) == 0:
            logger.warning(f"No images found in {self.config.data_dir}")
            return
        
        sample_images = random.sample(all_images, min(num_samples, len(all_images)))
        
        for i, img_path in enumerate(sample_images):
            logger.info(f"Processing image {i+1}/{len(sample_images)}: {img_path}")
            
            # Preprocess image
            img = tf.keras.preprocessing.image.load_img(
                img_path,
                target_size=self.config.params_image_size[:-1]
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            preprocess_fn = get_preprocess_input(self.config.params_model_name)
            img_array = preprocess_fn(img_array)

            last_conv_layer_name = get_last_conv_layer_name(self.config.params_model_name)
            heatmap = self.make_gradcam_heatmap(img_array, last_conv_layer_name=last_conv_layer_name)
            
            save_path = self.config.heatmap_dir / f"gradcam_{i}.jpg"
            self.save_and_display_gradcam(img_path, heatmap, str(save_path))
        
        logger.info(f"Grad-CAM explanations saved to {self.config.heatmap_dir}")
