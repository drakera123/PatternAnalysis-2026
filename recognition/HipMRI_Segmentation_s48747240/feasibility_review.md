# Feasibility Review: HipMRI 2D Prostate Segmentation

*[Your Name], s4874724 — COMP3710 Pattern Recognition*

## 1. User Need, Scope & Acceptance Criteria

The intended user is a radiotherapy treatment planning specialist, who
currently manually contours the prostate and surrounding organs-at-risk
(OARs) on each MRI slice — a time-consuming step in planning. This
prototype aims to propose draft 2D segmentation masks (prostate, bladder,
rectum, bone, body outline) for the specialist to review and adjust,
rather than replace manual contouring outright.

Acceptance criteria:
- Mean Dice similarity coefficient ≥ 0.75 on the prostate label on the
  held-out test split
- Model correctly identifies at least 3 of 5 manually-flagged boundary
  failure cases (e.g. apex/base slices) during the qualitative autopsy
- Peak GPU VRAM during training stays within Rangpur's available GPU
  memory allocation (to be confirmed once run on a GPU node)
- Per-slice inference latency suitable for interactive clinical review
  (target: under 1 second/slice)

## 2. Model Choice & Course Concepts

The chosen dataset is the HipMRI 2D prostate slice dataset
(`keras_slices_data`), already split by the course team into
train/validate/test folders on Rangpur. The initial model is a standard
2D U-Net (Ronneberger et al., 2015), using an encoder-decoder
architecture with skip connections to preserve spatial detail lost
during downsampling — a core course concept for dense pixel-wise
prediction tasks. This establishes an Easy-difficulty baseline and
verifies the data pipeline. The project will then scale to Normal (2D
Improved U-Net or CAN) and Hard (3D Improved U-Net or CAN3D on
downsampled 3D HipMRI volumes) difficulty, following the course's
recommended incremental pathway (Section 1.3), with the simpler 2D U-Net
retained as the baseline for the mandatory benchmarking comparison
(Section 2.2).

## 3. Preliminary Feasibility Evidence

The full pipeline (`dataset.py`, `modules.py`, `train.py`, `predict.py`)
has been implemented in TensorFlow/Keras and smoke-tested end-to-end
locally, using synthetic data matching the real HipMRI folder/filename
structure (`case_*`/`seg_*` naming, six pre-split folders):

- `train.py` successfully built a 2D U-Net (7,771,462 parameters),
  trained for 19 epochs with early stopping, and saved both model
  checkpoints and loss/Dice training curve plots.
- `predict.py` successfully loaded the saved model, ran inference across
  the test split, computed per-class Dice scores, and saved 5 qualitative
  comparison visualisations (input / ground truth / prediction /
  difference overlay).
- Dice scores on this synthetic smoke test were ≈0.17 (near-random for
  6 classes), as expected, since the synthetic data is random noise with
  no learnable structure — this confirms the architecture, training loop,
  and evaluation code are functioning correctly, not that the model
  performs well (that requires the real dataset on Rangpur).
- Real HipMRI folder structure has been confirmed on Rangpur:
  `keras_slices_train/validate/test` with matching
  `keras_slices_seg_train/validate/test` label folders, filenames
  `case_<id>_week_<n>_slice_<n>.nii.gz` / `seg_<id>_week_<n>_slice_<n>.nii.gz`.

Outstanding before full training: confirm exact label class count and
slice dimensions against real data, and run on an actual Rangpur GPU
node (not yet attempted).

## 4. Risks, Budget & Fallback

Key risks: (1) 3D volumetric training may exceed available GPU memory on
Rangpur, requiring patch-based or downsampled subvolumes; (2) severe
class imbalance (prostate occupies <2% of each slice) may require a
weighted or Dice-based loss rather than plain cross-entropy — the
current `combined_loss` (Dice + cross-entropy) is a first attempt at
this; (3) label class count and slice dimensions are assumed, not yet
confirmed, against the real data.

Compute budget: Rangpur interactive GPU sessions for development/smoke
testing, Slurm batch jobs for full training runs.

Next planned experiment: run `train.py` on an actual Rangpur GPU node
against real HipMRI data, starting with `EARLY_STOP = True` for a fast
real-data smoke test before committing to a full run.

Fallback: if the 3D Hard-tier model proves infeasible within the project
timeline, fall back to submitting the Normal-tier 2D Improved U-Net/CAN
on HipMRI, which remains a complete and defensible submission.
