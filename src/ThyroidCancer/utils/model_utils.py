import tensorflow as tf


def get_backbone(model_name: str, input_shape, weights: str, include_top: bool):
    name = (model_name or "").lower()
    if name in ("efficientnetb0", "effnetb0"):
        return tf.keras.applications.EfficientNetB0(
            input_shape=tuple(input_shape),
            weights=weights,
            include_top=include_top,
        )
    if name == "resnet50":
        return tf.keras.applications.ResNet50(
            input_shape=tuple(input_shape),
            weights=weights,
            include_top=include_top,
        )
    if name == "densenet121":
        return tf.keras.applications.DenseNet121(
            input_shape=tuple(input_shape),
            weights=weights,
            include_top=include_top,
        )
    if name == "mobilenetv2":
        return tf.keras.applications.MobileNetV2(
            input_shape=tuple(input_shape),
            weights=weights,
            include_top=include_top,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def get_preprocess_input(model_name: str):
    name = (model_name or "").lower()
    if name in ("efficientnetb0", "effnetb0"):
        return tf.keras.applications.efficientnet.preprocess_input
    if name == "resnet50":
        return tf.keras.applications.resnet.preprocess_input
    if name == "densenet121":
        return tf.keras.applications.densenet.preprocess_input
    if name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    raise ValueError(f"Unsupported model_name: {model_name}")


def add_classification_head(base_model: tf.keras.Model, classes: int) -> tf.keras.Model:
    pooling = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    dense1 = tf.keras.layers.Dense(512, activation="relu")(pooling)
    bn1 = tf.keras.layers.BatchNormalization()(dense1)
    dropout1 = tf.keras.layers.Dropout(0.5)(bn1)
    dense2 = tf.keras.layers.Dense(256, activation="relu")(dropout1)
    bn2 = tf.keras.layers.BatchNormalization()(dense2)
    dropout2 = tf.keras.layers.Dropout(0.3)(bn2)
    predictions = tf.keras.layers.Dense(
        classes,
        activation="softmax",
        name="output_layer",
    )(dropout2)

    return tf.keras.models.Model(inputs=base_model.input, outputs=predictions)


def get_last_conv_layer_name(model_name: str) -> str:
    name = (model_name or "").lower()
    if name in ("efficientnetb0", "effnetb0"):
        return "top_activation"
    if name == "resnet50":
        return "conv5_block3_out"
    if name == "densenet121":
        return "conv5_block16_concat"
    if name == "mobilenetv2":
        return "Conv_1"
    raise ValueError(f"Unsupported model_name: {model_name}")
