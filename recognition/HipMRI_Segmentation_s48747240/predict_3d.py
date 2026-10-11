"""
predict.py  (3D HipMRI)

Usage: python predict.py [baseline3d|improved3d]
Per-class Dice per test volume (only volumes where the class is present),
then axial-slice figures for the weakest class: 3 worst volumes, one partial
overlap (Dice closest to 0.4) and the best volume.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import load_hipmri_3d, NUM_CLASSES

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "improved3d"
CKPT = {"baseline3d": "unet3d", "improved3d": "improved3d"}[MODEL_NAME]
OUTPUT_DIR = f"predictions_{MODEL_NAME}"


def dice(true_lab, pred_lab, c, smooth=1e-6):
    t, p = true_lab == c, pred_lab == c
    return (2.0 * np.sum(t & p) + smooth) / (t.sum() + p.sum() + smooth)


def show(image, true_lab, pred_lab, name, k, cls, d, out_dir):
    z = int(np.round(np.argwhere(true_lab == cls)[:, 2].mean()))  # axial slice through GT centroid
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
    ax[0].imshow(image[:, :, z], cmap="gray"); ax[0].set_title("Input (axial)")
    ax[1].imshow(true_lab[:, :, z], cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1); ax[1].set_title("Ground truth")
    ax[2].imshow(pred_lab[:, :, z], cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1); ax[2].set_title("Prediction")
    ax[3].imshow((true_lab[:, :, z] != pred_lab[:, :, z]), cmap="Reds"); ax[3].set_title("Difference")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{name}  slice z={z}  class {cls} Dice={d:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"vol_{k:02d}_{name.split('_LFOV')[0]}.png"))
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = tf.keras.models.load_model(os.path.join("checkpoints", f"{CKPT}_best.keras"), compile=False)
    _, _, test_ds, names = load_hipmri_3d(1)

    imgs, trues, preds = [], [], []
    for x, y in test_ds:
        p = model(x, training=False).numpy()[0]
        imgs.append(x.numpy()[0, ..., 0])
        trues.append(np.argmax(y.numpy()[0], -1))
        preds.append(np.argmax(p, -1))

    D = np.array([[dice(t, p, c) for c in range(NUM_CLASSES)] for t, p in zip(trues, preds)])
    present = np.array([[np.any(t == c) for c in range(NUM_CLASSES)] for t in trues])

    print("\n--- Per-class Dice, test volumes ---")
    for c in range(NUM_CLASSES):
        if present[:, c].any():
            print(f"  Class {c}: where present {D[present[:, c], c].mean():.4f} "
                  f"(present in {present[:, c].sum()}/{len(D)} volumes, all volumes {D[:, c].mean():.4f})")
    fg = [D[present[:, c], c].mean() for c in range(1, NUM_CLASSES) if present[:, c].any()]
    print(f"  Mean foreground Dice (classes 1-5): {np.mean(fg):.4f}")

    cls = min((c for c in range(1, NUM_CLASSES) if present[:, c].any()),
              key=lambda c: D[present[:, c], c].mean())
    idx = np.where(present[:, cls])[0]
    order = idx[np.argsort(D[idx, cls])]
    chosen = list(order[:3])
    partial = idx[np.argmin(np.abs(D[idx, cls] - 0.4))]
    for extra in (partial, order[-1]):
        if extra not in chosen:
            chosen.append(extra)
    print(f"\nWeakest class: {cls}. Figures for volumes: {[names[i] for i in chosen]}")
    for k, i in enumerate(chosen):
        print(f"  {names[i]}: class {cls} Dice={D[i, cls]:.2f}")
        show(imgs[i], trues[i], preds[i], names[i], k, cls, D[i, cls], OUTPUT_DIR)


if __name__ == "__main__":
    main()