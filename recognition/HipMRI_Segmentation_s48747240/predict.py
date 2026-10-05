"""
predict.py

Example usage of the trained 2D U-Net: loads saved model weights, runs
inference on the test split, prints per-case Dice scores, and saves
visual comparisons (input / ground truth / prediction / difference overlay)
for qualitative inspection and failure-case autopsy.

NumPy is used here for visualization/array handling, as permitted by the
assignment spec for predict.py specifically.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from modules import dice_coefficient, combined_loss
from dataset import load_hipmri_2d

CHECKPOINT_PATH = "checkpoints/unet2d_best.keras"
OUTPUT_DIR = "predictions"
NUM_EXAMPLES_TO_VISUALIZE = 5


def load_trained_model(checkpoint_path: str) -> tf.keras.Model:
    """Load a saved model, registering the custom loss/metric used in training."""
    return tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "combined_loss": combined_loss,
        },
    )


def per_class_dice(y_true, y_pred, num_classes, smooth=1e-6):
    """Compute Dice score separately for each class channel."""
    scores = []
    for c in range(num_classes):
        true_c = y_true[..., c].flatten()
        pred_c = y_pred[..., c].flatten()
        intersection = np.sum(true_c * pred_c)
        denom = np.sum(true_c) + np.sum(pred_c)
        dice = (2.0 * intersection + smooth) / (denom + smooth)
        scores.append(dice)
    return scores


def visualize_prediction(image, true_mask, pred_mask, index, output_dir):
    """
    Save a 4-panel figure: input image, ground truth mask, predicted mask,
    and a difference overlay (false positives/negatives) for qualitative
    failure-case inspection.
    """
    true_labels = np.argmax(true_mask, axis=-1)
    pred_labels = np.argmax(pred_mask, axis=-1)
    difference = (true_labels != pred_labels).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image[..., 0], cmap="gray")
    axes[0].set_title("Input MRI Slice")
    axes[0].axis("off")

    axes[1].imshow(true_labels, cmap="tab10")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_labels, cmap="tab10")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    axes[3].imshow(difference, cmap="Reds")
    axes[3].set_title("Difference (errors in red)")
    axes[3].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"prediction_{index:03d}.png"))
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading trained model from {CHECKPOINT_PATH}...")
    model = load_trained_model(CHECKPOINT_PATH)

    print("Loading test data...")
    _, _, test_ds = load_hipmri_2d(batch_size=1, early_stop=False)

    print("Running inference on test set...")
    all_dice_scores = []
    example_count = 0

    for batch_idx, (images, true_masks) in enumerate(test_ds):
        pred_masks = model.predict(images, verbose=0)

        num_classes = true_masks.shape[-1]
        batch_dice = per_class_dice(
            true_masks.numpy()[0], pred_masks[0], num_classes
        )
        all_dice_scores.append(batch_dice)

        if example_count < NUM_EXAMPLES_TO_VISUALIZE:
            visualize_prediction(
                images.numpy()[0], true_masks.numpy()[0], pred_masks[0],
                example_count, OUTPUT_DIR,
            )
            example_count += 1

    all_dice_scores = np.array(all_dice_scores)  # (num_test_cases, num_classes)
    mean_per_class = all_dice_scores.mean(axis=0)

    print("\n--- Per-class mean Dice score on test set ---")
    for c, score in enumerate(mean_per_class):
        print(f"  Class {c}: {score:.4f}")
    print(f"  Overall mean Dice: {mean_per_class.mean():.4f}")

    print(f"\nSaved {example_count} prediction visualizations to {OUTPUT_DIR}/")
    print("Inspect these for the failure-case autopsy required by the report "
          "(3-5 representative cases, including boundary failures).")


if __name__ == "__main__":
    main()
