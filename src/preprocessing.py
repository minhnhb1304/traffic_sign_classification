"""Image preprocessing and data augmentation for traffic sign classification."""
import math

import tensorflow as tf


def _random_affine(img: tf.Tensor, max_deg: float = 10.0,
                   max_trans: float = 0.05, max_zoom: float = 0.08) -> tf.Tensor:
    """Randomly rotate/zoom/translate a rank-3 image tensor using tf.raw_ops.ImageProjectiveTransformV3."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    angle = tf.random.uniform([], -max_deg, max_deg) * math.pi / 180.0
    cos_a, sin_a = tf.cos(angle), tf.sin(angle)

    zoom = 1.0 + tf.random.uniform([], -max_zoom, max_zoom)
    tx = tf.random.uniform([], -max_trans, max_trans) * w
    ty = tf.random.uniform([], -max_trans, max_trans) * h

    cx, cy = w / 2.0, h / 2.0
    a0 = (cos_a / zoom)
    a1 = (sin_a / zoom)
    b0 = -(sin_a / zoom)
    b1 = (cos_a / zoom)
    a2 = cx - cx * a0 - cy * a1 + tx
    b2 = cy - cx * b0 - cy * b1 + ty

    transforms = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transforms,
        output_shape=tf.shape(img)[:2],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )
    return out[0]


def augment(img: tf.Tensor) -> tf.Tensor:
    """Augments a single rank-3 [0,1] image.

    Note: DO NOT flip horizontally (directional signs like 'Turn Left' would break).
    Uses pure tf.image.* and tf.raw_ops to ensure randomness works well inside tf.data.
    """
    img = _random_affine(img, max_deg=12.0, max_trans=0.06, max_zoom=0.10)
    img = tf.image.random_brightness(img, max_delta=0.12)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def build_augmentation_pipeline():
    """Returns a callable augment(img)->img for use in tf.data.Dataset.map."""
    return augment


def preprocess_single_image(img: tf.Tensor, img_size: int) -> tf.Tensor:
    """Preprocess a single image (used for inference)."""
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img
