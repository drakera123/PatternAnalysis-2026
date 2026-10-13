import numpy as np
import tensorflow as tf
from dataset_3d import load_volume, list_pairs

model = tf.keras.models.load_model("checkpoints/improved3d_best.keras", compile=False)

pairs = [p for p in list_pairs() if "T014_Week1" in p[0] or "T014_Week2" in p[0]]
for m, l in pairs:
    x, y = load_volume(m, l)  # x: image (128,128,64), y: labels
    print(m.split("/")[-1], "shape", x.shape)
    for ax in range(3):
        for side, idx in (("first", 0), ("last", -1)):
            sl = np.take(y, idx, axis=ax)
            sx = np.take(x, idx, axis=ax)
            print(f"  axis {ax} {side}: label values {np.unique(sl)}, "
                  f"nonzero label frac {np.mean(sl > 0):.3f}, img std {sx.std():.3f}")
    p = model(x[None, ..., None].astype("float32"), training=False).numpy()[0].argmax(-1)
    for ax in range(3):
        print(f"  pred classes, axis {ax} first/last:",
              np.unique(np.take(p, 0, axis=ax)), np.unique(np.take(p, -1, axis=ax)))

    d = (y != p)
    cols = np.where(d.any(axis=(0, 2)))[0]
    print("columns with disagreement along axis 1:", cols.min(), cols.max())
    print("disagreement per column (top 5):", np.argsort(d.sum(axis=(0, 2)))[-5:])
