# Feasibility Review: HipMRI 2D Prostate Segmentation

*Marcus Wee Lam Seow, s4874724 — COMP3710 Pattern Recognition*

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

## 3. Feasibility review

This is the review presented at the midpoint check-off. At that point the
plan was a 2D U-Net baseline with a path to Normal-tier (2D Improved
U-Net/CAN) and, time permitting, Hard-tier 3D. The project was later
extended to the 3D Improved U-Net and 3D U-Net baseline reported here.

- **User, scope and acceptance criteria.** User: a radiotherapy treatment
  planning specialist who currently manually contours the prostate and
  surrounding organs-at-risk (OARs) on each MRI slice. The prototype
  proposes draft 2D segmentation masks (prostate, bladder, rectum, bone,
  body outline) for the specialist to review and adjust, not to replace
  manual contouring. Criteria at the time: (1) mean Dice >= 0.75 on the
  prostate label on the held-out test split; (2) the model correctly
  identifies at least 3 of 5 manually-flagged boundary failure cases
  (e.g. apex/base slices) during the qualitative autopsy; (3) peak GPU
  VRAM during training stays within Rangpur's available allocation
  (to be confirmed once run on a GPU node); (4) per-slice inference
  latency suitable for interactive clinical review (target: under 1
  second/slice).
- **Model and course concepts.** HipMRI 2D prostate slice dataset
  (`keras_slices_data`), already split by the course team into
  train/validate/test folders on Rangpur. Initial model: a standard 2D
  U-Net (Ronneberger et al., 2015), encoder-decoder with skip connections
  to preserve spatial detail lost during downsampling. This establishes
  an Easy-difficulty baseline and verifies the data pipeline, with a
  planned path to Normal (2D Improved U-Net or CAN) and Hard (3D Improved
  U-Net or CAN3D on downsampled 3D HipMRI volumes), following the
  course's recommended incremental pathway, with the simpler 2D U-Net
  retained as the baseline for the mandatory benchmarking comparison.
- **Preliminary evidence.** The full pipeline (`dataset.py`, `modules.py`,
  `train.py`, `predict.py`) was implemented in TensorFlow/Keras and
  smoke-tested end-to-end locally on synthetic data matching the real
  HipMRI folder/filename structure. `train.py` built a 2D U-Net
  (7,771,462 parameters), trained for 19 epochs with early stopping, and
  saved checkpoints and training curves. `predict.py` loaded the saved
  model, ran inference across the test split, computed per-class Dice
  scores, and saved 5 qualitative comparison visualisations. Dice on the
  synthetic smoke test was ~0.17 (near-random for 6 classes), as
  expected, confirming the architecture, training loop and evaluation
  code worked correctly, not that the model performed well. The real
  HipMRI folder structure was confirmed on Rangpur:
  `keras_slices_train/validate/test` with matching
  `keras_slices_seg_train/validate/test` label folders.
- **Risks and fallback (at the time).** 3D volumetric training might
  exceed available GPU memory, requiring patch-based or downsampled
  subvolumes; severe class imbalance (prostate occupies <2% of each
  slice) might require a weighted or Dice-based loss rather than plain
  cross-entropy; label class count and slice dimensions were assumed,
  not yet confirmed. Fallback at the time: if the 3D Hard-tier model
  proved infeasible, submit the Normal-tier 2D Improved U-Net/CAN
  instead. In the event, the 3D Hard-tier models (Section 1) were
  successfully trained and are the final submission; the earlier 2D work
  is preserved in `2d_u-net` as supporting history.

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
