"""
dataset.py

Data loading and preprocessing for the HipMRI 2D prostate segmentation task.

The course has already split the data into six fixed folders under
/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/:

    keras_slices_train       / keras_slices_seg_train
    keras_slices_validate    / keras_slices_seg_validate
    keras_slices_test        / keras_slices_seg_test

Image files are named case_<id>_week_<n>_slice_<n>.nii.gz and their matching
label files are named seg_<id>_week_<n>_slice_<n>.nii.gz (same suffix, prefix
swapped). Because the splits are pre-made by the course team, we load each
folder directly rather than re-splitting by patient ourselves - we only run
a sanity check to confirm no case id leaks across splits.

NumPy is used here (and only here / in predict.py) for I/O and preprocessing,
per the assignment spec which restricts modules.py to pure TF/Keras.
"""

import os
import re
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm

BASE_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"


SPLIT_FOLDERS = {
    "train": ("keras_slices_train", "keras_slices_seg_train"),
    "validate": ("keras_slices_validate", "keras_slices_seg_validate"),
    "test": ("keras_slices_test", "keras_slices_seg_test"),
}


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """One-hot encode a label mask into (H, W, num_classes)."""
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res


def load_data_2D(image_names, norm_image=False, categorical=False,
                  dtype=np.float32, early_stop=False):
    """Load a list of 2D Nifti files into a single pre-allocated array."""
    num = len(image_names)
    first_case = nib.load(image_names[0]).get_fdata(caching="unchanged")
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, name in enumerate(tqdm(image_names)):
        nifti_image = nib.load(name)
        in_image = nifti_image.get_fdata(caching="unchanged")
        if len(in_image.shape) == 3:
            in_image = in_image[:, :, 0]
        in_image = in_image.astype(dtype)
        if norm_image:
            in_image = (in_image - in_image.mean()) / (in_image.std() + 1e-8)
        if categorical:
            in_image = to_channels(in_image, dtype=dtype)
            images[i, :, :, :] = in_image
        else:
            images[i, :, :] = in_image
        if early_stop and i > 20:
            break

    return images


def extract_case_id(filepath: str) -> str:
    """Extract 'case_004' style id from a filename, for leakage sanity checks."""
    match = re.search(r"(case|seg)_(\d+)", os.path.basename(filepath))
    if match is None:
        raise ValueError(f"Could not extract case id from {filepath}")
    return match.group(2)  # just the numeric id, shared between case_/seg_ prefixes


def check_no_leakage(train_files, val_files, test_files):
    """
    Sanity check only (the splits are pre-made by the course): confirms no
    case id appears in more than one split. Raises if leakage is found.
    """
    train_ids = set(extract_case_id(f) for f in train_files)
    val_ids = set(extract_case_id(f) for f in val_files)
    test_ids = set(extract_case_id(f) for f in test_files)

    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap:
        raise ValueError(f"Data leakage detected - case ids in multiple splits: {overlap}")


def _image_to_seg_path(image_path: str) -> str:
    """Map a case_*.nii.gz image path to its matching seg_*.nii.gz label path."""
    image_dir = os.path.dirname(image_path)
    seg_dir = image_dir.replace("keras_slices_", "keras_slices_seg_", 1) \
        if "keras_slices_seg_" not in image_dir else image_dir
    filename = os.path.basename(image_path).replace("case_", "seg_", 1)
    return os.path.join(seg_dir, filename)


def build_tf_dataset(images, masks, batch_size=16, shuffle=True, augment=False):
    """Wrap numpy arrays into a batched tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=42)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _augment(image, mask):
    """Simple flip augmentation; extend with rotations/intensity jitter as needed."""
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask


def _load_split(split: str, early_stop: bool = False):
    """Load one of 'train' / 'validate' / 'test' into (images, masks) numpy arrays."""
    image_folder, _ = SPLIT_FOLDERS[split]
    image_paths = sorted(glob.glob(os.path.join(BASE_DIR, image_folder, "*.nii.gz")))
    mask_paths = [_image_to_seg_path(p) for p in image_paths]

    x = load_data_2D(image_paths, norm_image=True, early_stop=early_stop)
    y = load_data_2D(mask_paths, categorical=True, dtype=np.uint8, early_stop=early_stop)

    x = x[..., np.newaxis].astype(np.float32)  # add channel dim
    y = y.astype(np.float32)
    return x, y, image_paths


def load_hipmri_2d(batch_size: int = 16, early_stop: bool = False):
    """
    Loads the pre-split HipMRI 2D train/validate/test folders, runs a
    leakage sanity check, and returns three tf.data.Dataset objects.
    """
    x_train, y_train, train_paths = _load_split("train", early_stop)
    x_val, y_val, val_paths = _load_split("validate", early_stop)
    x_test, y_test, test_paths = _load_split("test", early_stop)

    check_no_leakage(train_paths, val_paths, test_paths)

    train_ds = build_tf_dataset(x_train, y_train, batch_size, shuffle=True, augment=True)
    val_ds = build_tf_dataset(x_val, y_val, batch_size, shuffle=False)
    test_ds = build_tf_dataset(x_test, y_test, batch_size, shuffle=False)

    return train_ds, val_ds, test_ds
