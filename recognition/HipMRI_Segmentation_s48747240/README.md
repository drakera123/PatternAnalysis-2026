# HipMRI Prostate MRI Segmentation: 3D Improved U-Net (Hard) with 3D U-Net baseline

*COMP3710 Pattern Recognition Report, Marcus Wee Lam Seow, s48747240*

**Selected model: 3D Improved U-Net on downsampled HipMRI 3D volumes (Hard difficulty).** A plain 3D U-Net is used as the baseline for comparison.

## 1. Project description

Radiotherapy planning needs accurate contours of the prostate and the organs around it. This project segments pelvic MRI volumes from the HipMRI study into six labels (label 0 plus five anatomical structures). The task target is Dice >= 0.7 on the test set.

**How it works.** The 3D Improved U-Net follows Isensee et al. (2018), adapted to 3D:
- pre-activation residual *context modules* (two 3x3x3 convolutions with spatial dropout 0.3 between them and a skip connection);
- instance normalisation (GroupNormalization with one group per channel) and leaky ReLU (slope 0.01) in place of batch norm and ReLU;
- strided 3x3x3 convolutions for downsampling (five levels, 16 to 256 filters);
- *localization modules* in the decoder (3x3x3 then 1x1x1 convolution);
- deep supervision: segmentation outputs from the three highest-resolution decoder levels are upsampled and summed before the softmax, which keeps a single output.

It has 8,267,618 parameters. The baseline is a standard 3D U-Net (Cicek et al., 2016): four levels (16 to 128 filters), a 256-filter bottleneck, max pooling, transposed-convolution upsampling, batch norm and ReLU (5,650,886 parameters). Both models use the same data, split, loss and optimiser, and are evaluated on the same 27 test volumes.

**Loss and optimiser.** Mean per-class soft Dice loss plus categorical cross-entropy, Adam (lr 5e-4, halved on validation-loss plateau), batch size 2, up to 100 epochs, early stopping (patience 20) and best-checkpoint saving on validation foreground Dice. The per-class Dice loss is used instead of a global Dice term because a global term is dominated by background voxels, which would otherwise swamp the signal from small structures.

![Training curves, 3D Improved U-Net](plots/training_curves_improved3d.png)
![Training curves, 3D U-Net baseline](plots/training_curves_unet3d.png)

## 2. Data use and acknowledgments

This project uses data from the **HipMRI Study**, accessed via the University of Queensland's Rangpur HPC cluster (`/home/groups/comp3710/HipMRI_Study_open/`), under the terms of the HipMRI Study data use agreement.

**Acknowledgments:** Data were provided in part by the HipMRI Study.

**Citation:** The agreement leaves the citation for image segmentation blank (`data_use_agreement.txt`). This has been raised with the teaching team and will be updated once confirmed. In the interim, the following publications listed in the agreement are cited:

- HipMRI Study for MRI-alone treatment planning and other imaging: https://doi.org/10.1016/j.ijrobp.2015.08.045
- HipMRI Study for image synthesis and translation: https://arxiv.org/abs/2011.13615
- HipMRI Study for image segmentation: *[citation pending, confirm with teaching team]*

**Data use terms acknowledged:** no attempt has been made to identify or contact any participant; the data and derived results are used for non-commercial coursework only; no warranty is made regarding the correctness of the data.

## 3. Feasibility review

**Purpose.** To decide, before committing GPU time, whether a 3D Improved U-Net on downsampled HipMRI volumes was feasible and how success would be judged.

**User need and scope.** The user is a radiotherapy treatment planning specialist who contours the prostate and surrounding organs-at-risk on each MRI volume by hand. The prototype proposes draft 3D segmentation masks (six labels) for the specialist to review and edit, not to replace manual contouring. The task is limited to downsampled volumes (128x128x64) from the HipMRI semantic MR and label folders.

**Acceptance criteria.**
1. Dice >= 0.7 on every foreground class on a held-out test set.
2. Strict patient-level separation: every scan of a patient lies in exactly one of train, validation and test, enforced by an assertion.
3. Per-class Dice reported per test volume, with 3 to 5 failure cases inspected visually.
4. Resource limits: peak training memory within one A100 (40 GB), inference under 1 second per volume, and a full training run within a few GPU-hours.
5. The Improved U-Net compared against a plain 3D U-Net on the same split.

**Model choice and course concepts.** The model is a 3D Improved U-Net (Isensee et al., 2018), which is Hard difficulty: residual context modules, instance normalisation with leaky ReLU, strided-convolution downsampling, localization modules and deep supervision. A plain 3D U-Net (Cicek et al., 2016) with skip connections is the baseline. The Dice-plus-cross-entropy loss is computed per class, because small structures such as class 5 occupy about 0.1% of voxels and a global Dice would be dominated by background. Downsampling by 2 in each axis keeps the volumes within GPU memory.

**Preliminary feasibility evidence.**
- *Data audit.* 211 MR volumes pair one-to-one with 211 label volumes from 38 patients. Voxel spacing is about 1.56 mm in all axes. 210 volumes are 256x256x128 and one is 256x256x144 (centre-cropped to 128). Labels take values 0 to 5. In the first label volume, class 5 covers about 0.12% of voxels.
- *Model build.* Both models build with output shape (128, 128, 64, 6): 5,650,886 parameters (baseline) and 8,267,618 parameters (Improved U-Net).
- *Smoke test.* A 16-volume, 2-epoch run on one A100 trained at about 1 s per step with batch size 2 (after a 87 s first-epoch compile), with peak GPU memory of 3.32 GB. Loss fell from 2.02 to 1.02 and mean foreground Dice rose from 0.11 to 0.21.
- *Time estimate.* About 70 steps per epoch at roughly 0.1 to 0.15 s per step puts a full run at well under a GPU-hour, even for 100 epochs.

**Risks, budget and fallback.**

| Risk | Mitigation |
|---|---|
| Few patients (38), so a small, noisy test set | Patient-level split; report per-class Dice and state the limitation |
| Severe class imbalance for small structures | Per-class Dice loss plus cross-entropy |
| Downsampling removes detail of small structures | Nearest-neighbour label sampling; state that Dice is at reduced resolution |
| Memory pressure and node crashes seen on the cluster | Images stored as float16, labels as uint8, one-hot encoding per batch; a crashing node excluded |
| Long queue waits | Smoke test first, short time limits, dependent prediction jobs |

*Compute budget:* one A100 on partition `comp3710`; a 20-minute smoke job, then one training job and one prediction job per model. *Next planned experiment:* the full 100-epoch run of both models with early stopping. *Fallback:* if the Improved U-Net did not train reliably, report the plain 3D U-Net as the delivered model and describe the Improved U-Net as incomplete.

## 4. Data pipeline

Data: `/home/groups/comp3710/HipMRI_Study_open/semantic_MRs` and `semantic_labels_only` (not committed). `B006_Week0_LFOV.nii.gz` pairs with `B006_Week0_SEMANTIC.nii.gz`.

- **Patient-level split.** The first part of the filename (for example `B006`) is the patient ID, and each patient has between 1 and 8 weekly scans. Patients are shuffled with a fixed seed (42) and every scan of a patient goes to exactly one split, with an assertion that the splits are disjoint. The split is 26 / 6 / 6 patients, which gives **136 training, 48 validation and 27 test volumes**. The test patients are J026, K008, S028, T009, T014 and V027. Validation drives early stopping and checkpointing; the test set is used only for final evaluation. I treat the letter-and-number prefix (for example `B006`) as one patient, because every prefix has weekly scans (Week0 to Week7) following the same naming pattern. This is an inference from the filenames and was not confirmed against dataset documentation.
- **Depth fix.** One volume has 144 slices in depth; it is centre-cropped to 128.
- **Downsampling.** 256x256x128 to 128x128x64. Images use 2x2x2 block averaging; labels use strided (nearest-neighbour) sampling so label values stay valid. Class 5 is about 0.1% of voxels at full resolution, so small structures lose detail in this step.
- **Intensity normalisation.** Each volume is z-scored.
- **Memory.** Images are held as float16 and labels as uint8; one-hot encoding happens per batch in the input pipeline. (An earlier version held float32 one-hot arrays in RAM and crashed one node.)
- **Augmentation.** Random flip along axis 0 on the training set only. I assumed axis 0 is the left-right axis. In the prediction slices I inspected, the anatomy appears bilaterally symmetric and consistently oriented, so I saw no sign that the flip created mirrored or misoriented training cases. I did not verify the axis from the NIfTI affine, so this is a visual check only.
## 5. Dependencies and reproducibility

Python 3.11 (conda env `tf`), tensorflow 2.21.0, nibabel 5.4.2, numpy 2.4.6, matplotlib 3.11.2, tqdm 4.70.1. Run on one NVIDIA A100 40 GB on Rangpur, partition `comp3710`, with no `--mem` request (the nodes reject one).

```bash
sbatch train_3d_job.sh            # 3D Improved U-Net -> checkpoints/improved3d_best.keras
sbatch train_3d_baseline_job.sh   # 3D U-Net baseline -> checkpoints/unet3d_best.keras
sbatch predict_3d_job.sh          # figures in predictions_improved3d/
sbatch predict_3d_baseline_job.sh # figures in predictions_baseline3d/
```

The data split and training shuffle use seed 42, but no global TensorFlow seed is set and GPU kernels are nondeterministic, so reruns give slightly different numbers. Checkpoints are not committed.

| File | Purpose |
|---|---|
| `modules_3d.py` | 3D U-Net baseline, 3D Improved U-Net, per-class Dice loss and metric |
| `dataset_3d.py` | Pairing, patient-level split, resizing, normalisation, batching |
| `train_3d.py` | Training, validation, test evaluation, curves, resource profiling |
| `predict_3d.py` | Per-class Dice per test volume and failure-case figures |

## 6. Results

All numbers are from one training run per model. Dice is computed per test volume and averaged, at the 128x128x64 working resolution. Every class is present in all 27 test volumes.

### 6.1 Per-class Dice (27 test volumes, 6 patients)

| Class | 3D Improved U-Net | 3D U-Net baseline |
|---|---|---|
| 0 | 0.998 | 0.998 |
| 1 | 0.986 | 0.987 |
| 2 | 0.912 | 0.921 |
| 3 | 0.931 | 0.946 |
| 4 | 0.842 | 0.847 |
| 5 | 0.817 | 0.837 |
| **Mean foreground (1 to 5)** | **0.897** | **0.908** |

**Target (Dice >= 0.7):** both models exceed 0.7 on every class, including the weakest (class 5: 0.82 and 0.84). The prostate label was not confirmed against dataset documentation, so results are reported by class number; the target is met on all of them.

The Improved U-Net did **not** beat the plain 3D U-Net: its mean foreground Dice is about 0.01 lower. With single runs on 6 test patients I treat this as a tie, not a loss, and I make no claim about which architecture is better.

### 6.2 Resource profile

| | 3D Improved U-Net | 3D U-Net baseline |
|---|---|---|
| Parameters | 8,267,618 | 5,650,886 |
| Peak GPU memory (training) | 3.93 GB | 3.26 GB |
| Inference per volume | 119.9 ms | 49.6 ms |
| Epochs run | 76 of 100 (early stopping; best epoch 56) | 100 of 100 |
| Training step time | about 90 to 135 ms | about 66 to 100 ms |
| Approx. training time | ~11.9 min (76 epochs) | ~11.3 min (100 epochs) |

The Improved U-Net has 1.46x the parameters of the baseline but is about 2.4x slower at inference, because instance normalisation and the extra high-resolution decoder convolutions are expensive.

### 6.3 Failure cases

`predict_3d.py` scores every test volume, finds the weakest class (class 5 for both models), and plots axial slices through the ground-truth region for the three lowest-scoring volumes and the best volume. Figures show input, ground truth, prediction and a difference overlay. Colours are fixed across figures.

| Model | Volume | Class 5 Dice | Figure |
|---|---|---|---|
| Improved | J026_Week1 | 0.70 | `predictions_improved3d/vol_00_J026_Week1.png` |
| Improved | J026_Week6 | 0.71 | `predictions_improved3d/vol_01_J026_Week6.png` |
| Improved | T014_Week1 | 0.75 | `predictions_improved3d/vol_02_T014_Week1.png` |
| Improved | V027_Week7 | 0.91 (best) | `predictions_improved3d/vol_03_V027_Week7.png` |
| Baseline | T014_Week2 | 0.75 | `predictions_baseline3d/vol_00_T014_Week2.png` |
| Baseline | J026_Week1 | 0.77 | `predictions_baseline3d/vol_01_J026_Week1.png` |
| Baseline | K008_Week0 | 0.78 | `predictions_baseline3d/vol_02_K008_Week0.png` |
| Baseline | J026_Week7 | 0.90 (best) | `predictions_baseline3d/vol_03_J026_Week7.png` |

![J026_Week1, Improved](predictions_improved3d/vol_00_J026_Week1.png)
![J026_Week6, Improved](predictions_improved3d/vol_01_J026_Week6.png)
![T014_Week1, Improved](predictions_improved3d/vol_02_T014_Week1.png)
![V027_Week7, Improved (best)](predictions_improved3d/vol_03_V027_Week7.png)
![T014_Week2, Baseline](predictions_baseline3d/vol_00_T014_Week2.png)
![J026_Week1, Baseline](predictions_baseline3d/vol_01_J026_Week1.png)

Across the six embedded failure figures (four Improved U-Net, two baseline), 
errors fall into two distinct patterns rather than a diffuse
blur:

1. **Boundary leakage on the paired lateral structures** (visible top and
   bottom of each slice): a thin, broken ring of disagreement traces the
   edge of an otherwise correctly detected region. The structure's core is
   right in both ground truth and prediction; only a 1-2 voxel margin at
   its edge is inconsistent.
2. **A denser error cluster at the interface between the two central
   structures** (likely adjacent soft-tissue organs in the prostate region).
   This is where most of the error mass concentrates in every weaker case.
   The input MRI shows little to no intensity contrast at this boundary,
   unlike the sharp edges around bone, so the model has little signal to
   place the boundary precisely.

T014_Week1 and T014_Week2 additionally show a thin vertical line of disagreement on the right side of the difference panel. I checked the outermost voxel layers: ground-truth labels, image and prediction are all zero/background there, and disagreement along the left-right axis is confined to columns 24 to 98 (Week1) and 27 to 96 (Week2), with the right-hand peak at columns 98 and 96. It is therefore not a crop, resize or padding artefact. It lies at an internal boundary, most likely a one- or two-voxel offset in the placement of a structure’s edge, but I did not trace which structure.

The best-performing volume (V027_Week7, Dice 0.91) shows the same two
failure patterns at much smaller scale, suggesting these are a matter of
degree rather than a different failure mode entirely.

Anatomy in all six inspected slices is bilaterally symmetric and
consistent with correctly oriented axial pelvic imaging, so the
left-right flip augmentation (Section 4) does not appear to have
introduced any mirrored or misoriented training cases.

Observations supported by the numbers alone:
- The three worst class-5 volumes per model score 0.70 to 0.78. These are mild shortfalls rather than complete misses.
- J026 contributes the worst volumes for both models (weeks 1 and 6 for the Improved U-Net, week 1 for the baseline), so the weakest results cluster on one patient.
- Both models fail on overlapping volumes (J026_Week1), which suggests the difficulty comes from the data (small structure, low contrast) more than from the architecture.

### 6.4 Limitations

- The test set is 27 volumes from 6 patients (several have 8 weekly scans of the same anatomy), so the effective sample is closer to 6 than 27. Differences of about 0.01 are within what I would expect from run-to-run variation.
- Dice is measured at 128x128x64 against strided-downsampled labels, so it is probably optimistic for small structures and is not full-resolution accuracy.
- Single run per model, no confidence intervals, no cross-validation.
- Only the worst three volumes and the best volume per model were inspected.

## 7. Engineering recommendation

The 3D models meet the Dice >= 0.7 target on every class, so a 3D model is a reasonable basis for an annotation assistant that proposes contours for a clinician to review and edit. I would not use it unsupervised yet:
1. **Prefer the simpler model.** The plain 3D U-Net matches the Improved U-Net on every class (about +0.01 mean foreground Dice) at about 2.4x faster inference and 0.7x the parameters, so the added complexity is not justified by this evidence.
2. **Review small structures.** Flag volumes where class 5 is small or has low model confidence for manual review, since the weakest volumes cluster on particular patients.
3. **Validate at full resolution.** Results here are at half resolution in each axis with strided labels; re-evaluate at full resolution before any clinical claim.
4. **More patients.** Use cross-validation over patients, or a larger held-out set, to quantify the variation between patients.

## 8. AI Usage Disclosure

Generative AI (Claude, Anthropic) was used during this project as a coding
and writing assistant, under my direction and with all outputs verified
against real runs on Rangpur before being included here. Specific uses:

- **Architecture implementation.** Claude helped translate the Isensee et
  al. (2018) Improved U-Net and the Cicek et al. (2016) 3D U-Net
  descriptions into TensorFlow/Keras code (`modules_3d.py`), including the
  context modules, strided downsampling, localization modules and deep
  supervision for the Improved U-Net. All architectural choices were
  specified by me (filter counts, dropout rate, normalisation choice); the
  code was reviewed and the parameter counts and output shapes confirmed
  against the real `model.summary()` output in `train_651650.out` and
  `train_651651.out` before being reported.
- **Data pipeline.** Claude assisted with the patient-level train/val/test
  split logic and leakage assertions in `dataset_3d.py`, and with
  diagnosing the real file-naming convention
  (`<PatientID>_Week<N>_LFOV.nii.gz` / `_SEMANTIC.nii.gz`) from directory
  listings I provided.
- **Evaluation and reporting.** Claude helped write the per-class Dice
  scoring and failure-case selection logic in `predict_3d.py`, and helped
  draft this README. All numbers quoted (Dice scores, parameter counts,
  GPU memory, training times, epoch counts) were taken directly from real
  Slurm job logs (`train_651650.out`, `train_651651.out`) that I pasted in
  and that Claude did not generate. The failure-case descriptions in
  Section 6.3 were written from my own visual inspection of the saved
  prediction figures, which Claude does not have independent access to.
- **Debugging.** Claude helped interpret Slurm/CUDA error messages and git
  workflow issues (merge conflicts, diverged branches) encountered while
  running jobs and syncing code between my laptop and Rangpur.

No AI tool was used to generate or fabricate experimental results; every
number in Sections 6 and the resource profile comes from an actual
training/evaluation run on Rangpur, and figures are the real output of
`predict_3d.py`.
