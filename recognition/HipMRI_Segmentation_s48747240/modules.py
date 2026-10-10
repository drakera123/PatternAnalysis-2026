"""
modules.py

Model components for HipMRI prostate segmentation.

Contains a standard 2D U-Net, used as:
  (a) the Easy-difficulty deliverable while the pipeline is first verified, and
  (b) the baseline model for the mandatory benchmarking comparison against
      the final (Hard-difficulty) 3D model.

Pure TensorFlow/Keras only - no NumPy dependency, per assignment constraints.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


def conv_block(x, filters, name_prefix, use_bn=True):
    """Two 3x3 convolutions, each with optional batch norm, then ReLU."""
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv1")(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv2")(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
    return x


def encoder_block(x, filters, name_prefix, use_bn=True):
    """Conv block followed by 2x2 max pooling; returns (skip, pooled)."""
    skip = conv_block(x, filters, name_prefix, use_bn)
    pooled = layers.MaxPooling2D(2, name=f"{name_prefix}_pool")(skip)
    return skip, pooled


def decoder_block(x, skip, filters, name_prefix, use_bn=True):
    """Transposed conv upsample, concat with skip connection, then conv block."""
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same",
                                name=f"{name_prefix}_upconv")(x)
    x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip])
    x = conv_block(x, filters, name_prefix, use_bn)
    return x


def build_unet_2d(input_shape=(256, 256, 1), num_classes=6, base_filters=32,
                  use_bn=True, name="UNet2D"):
    """
    Standard 2D U-Net (Ronneberger et al., 2015) for multi-class segmentation.

    Args:
        input_shape: (H, W, C) of input MRI slices.
        num_classes: number of segmentation label classes (one-hot channels).
        base_filters: number of filters in the first encoder block; doubles
            at each depth level as is standard for U-Net.

    Returns:
        A compiled-free tf.keras.Model. Compile it in train.py with your
        chosen loss (e.g. Dice loss or categorical crossentropy + Dice) and
        optimizer.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Encoder
    skip1, pool1 = encoder_block(inputs, base_filters, "enc1", use_bn)
    skip2, pool2 = encoder_block(pool1, base_filters * 2, "enc2", use_bn)
    skip3, pool3 = encoder_block(pool2, base_filters * 4, "enc3", use_bn)
    skip4, pool4 = encoder_block(pool3, base_filters * 8, "enc4", use_bn)

    # Bottleneck
    bottleneck = conv_block(pool4, base_filters * 16, "bottleneck", use_bn)

    # Decoder
    up4 = decoder_block(bottleneck, skip4, base_filters * 8, "dec4", use_bn)
    up3 = decoder_block(up4, skip3, base_filters * 4, "dec3", use_bn)
    up2 = decoder_block(up3, skip2, base_filters * 2, "dec2", use_bn)
    up1 = decoder_block(up2, skip1, base_filters, "dec1", use_bn)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax",
                             name="segmentation_output")(up1)

    return Model(inputs, outputs, name=name)


def build_baseline_2d(input_shape=(256, 128, 1), num_classes=6):
    """Reduced baseline: half the filters (16 vs 32) and no batch norm."""
    return build_unet_2d(input_shape, num_classes, base_filters=16,
                         use_bn=False, name="Baseline2D")


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Per-batch Dice coefficient averaged across classes; used as a metric."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    """1 - Dice coefficient, usable as a standalone or combined loss term."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """Dice loss + categorical crossentropy, a common choice for segmentation."""
    cce = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    return dice_loss(y_true, y_pred) + cce
