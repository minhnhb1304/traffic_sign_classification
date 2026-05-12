"""Định nghĩa kiến trúc CNN tự xây cho bài toán phân loại biển báo VN."""
import tensorflow as tf
from tensorflow.keras import Model, layers

from . import config as C


def build_cnn(num_classes: int,
              img_size: int = C.IMG_SIZE,
              channels: int = C.IMG_CHANNELS) -> Model:
    """Kiến trúc CNN baseline: 3 khối Conv + Dense head.

    Khối 1: 32 filters  -> Khối 2: 64 -> Khối 3: 128.
    Mỗi khối: Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> ReLU -> MaxPool -> Dropout.
    """
    inputs = layers.Input(shape=(img_size, img_size, channels), name="image_input")

    x = _conv_block(inputs, filters=32, dropout=0.20, name="block1")
    x = _conv_block(x, filters=64, dropout=0.25, name="block2")
    x = _conv_block(x, filters=128, dropout=0.30, name="block3")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Dropout(0.5, name="fc1_drop")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name=C.MODEL_NAME)
    return model


def _conv_block(x, filters: int, dropout: float, name: str):
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    x = layers.MaxPooling2D(pool_size=2, name=f"{name}_pool")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def compile_model(model: Model, learning_rate: float = C.LEARNING_RATE) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    return model
