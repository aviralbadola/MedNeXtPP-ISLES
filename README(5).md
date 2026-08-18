# MedNeXt++ for Ischemic Stroke Lesion Segmentation

Official implementation of **MedNeXt++**, a 3D medical image segmentation framework developed for ischemic stroke lesion segmentation on the **ISLES** dataset.

The repository provides the complete research pipeline, including:

- Atlas-centric preprocessing
- MedNeXt++ model architecture
- nnU-Net v2 training integration
- MNI-space inference
- Inverse transformation to native patient space
- Native-space prediction visualization

---

## Overview

MedNeXt++ extends the MedNeXt architecture with additional mechanisms for capturing local, multi-scale, and global contextual information.

The proposed architecture incorporates:

- **LayerScale**
- **Dilated Large-Kernel Attention (DLKA)**
- **Global Context modeling**
- **Cross-Attention Fusion**

The model is integrated into the **nnU-Net v2** framework for 3D ischemic stroke lesion segmentation.

### Complete Pipeline

```text
Raw MRI + Lesion Mask
          │
          ▼
┌─────────────────────────────┐
│ Atlas-Centric Preprocessing │
│                             │
│ • Affine registration       │
│ • N4 bias correction        │
│ • Percentile normalization  │
└─────────────────────────────┘
          │
          ├── MRI → MNI space
          └── Mask → MNI space
          │
          ▼
     MNI-space Dataset
          │
          ▼
┌─────────────────────────────┐
│         nnU-Net v2          │
│                             │
│        MedNeXt++ (B)        │
└─────────────────────────────┘
          │
          ▼
    MNI-space Prediction
          │
          ▼
   Inverse Affine Transform
          │
          ▼
  Native-space Prediction
          │
          ▼
      Visualization
```

---

# 1. Method

## 1.1 MedNeXt++

MedNeXt++ is implemented as a 3D encoder-decoder segmentation network based on the MedNeXt architecture.

The implementation contains:

- Depthwise 3D convolutions
- Channel expansion and projection
- GELU activation
- Residual connections
- LayerScale
- Dilated Large-Kernel Attention (DLKA)
- Global Context modeling
- Cross-Attention Fusion
- Deep supervision

The architecture is implemented in:

```text
model/
├── blocks.py
├── MedNextPP.py
└── create_mednextpp.py
```

## 1.2 Dilated Large-Kernel Attention

DLKA is introduced to increase the effective receptive field while retaining the efficiency of depthwise convolutions.

The implemented module combines:

1. A depthwise 3D convolution with kernel size 5
2. A dilated depthwise 3D convolution with kernel size 7 and dilation 3
3. Instance normalization
4. Pointwise projection
5. Sigmoid gating

DLKA is incorporated at deeper encoder and bottleneck stages of MedNeXt++.

## 1.3 Global Context Modeling

Global Context modules provide additional global feature information at deeper levels of the encoder.

The implementation uses global average pooling followed by a lightweight channel-wise transformation and gating operation.

## 1.4 Cross-Attention Fusion

Cross-Attention Fusion is used in the decoder to combine encoder skip features with decoder features before subsequent decoding operations.

---

# 2. Model Configuration

The reported MedNeXt++ configuration uses the **B** model variant.

| Parameter | Value |
|---|---|
| Model | MedNeXt++ |
| Model ID | B |
| Dimension | 3D |
| Base channels | 32 |
| Kernel size | 3 |
| Deep supervision | Enabled |

The corresponding configuration is:

```python
n_channels = 32

exp_r = [2, 3, 4, 4, 4, 4, 4, 3, 2]

block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]
```

The model factory is implemented in:

```text
model/create_mednextpp.py
```

and the nnU-Net integration is implemented in:

```text
training/nnUNetTrainer_MedNeXtPP.py
```

---

# 3. Preprocessing

The preprocessing stage converts native patient data into a common MNI/atlas space before model training.

Implementation:

```text
preprocessing/atlas_preprocessing.py
```

## 3.1 MRI preprocessing

```text
Native MRI
    │
    ▼
Affine Registration to MNI152
    │
    ▼
N4 Bias-Field Correction
    │
    ▼
Percentile Intensity Normalization
    │
    ▼
MNI-space MRI
```

The current preprocessing configuration uses:

- Affine registration
- N4 bias-field correction
- N4 maximum iterations: 100
- Percentile normalization from the 0th to the 100th percentile
- Normalized intensity range: 0 to 1

## 3.2 Lesion-mask preprocessing

```text
Native Lesion Mask
        │
        ▼
Same Spatial Transformation
        │
        ▼
Nearest-Neighbor Interpolation
        │
        ▼
MNI-space Lesion Mask
```

Nearest-neighbor interpolation is used for lesion masks so that discrete segmentation labels are preserved.

The affine transformation generated during preprocessing is retained for subsequent inverse transformation of model predictions back to native patient space.

---

# 4. Installation

Clone the repository:

```bash
git clone https://github.com/aviralbadola/MedNeXtPP-ISLES.git
cd MedNeXtPP-ISLES
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The dependency list is maintained in:

```text
requirements.txt
```

---

# 5. Dataset Preparation

The dataset itself is **not included** in this repository.

Users must obtain the appropriate ISLES dataset separately and comply with the dataset's terms of use.

The preprocessing pipeline expects subject-wise native MRI and lesion-mask files together with the MNI152 atlas/template.

The exact raw-data organization should follow the structure expected by:

```text
preprocessing/atlas_preprocessing.py
```

---

# 6. Atlas-Centric Preprocessing

Run:

```bash
python preprocessing/atlas_preprocessing.py \
    --dataset-root /path/to/raw_dataset \
    --atlas /path/to/MNI152_T1_1mm.nii.gz \
    --output-root /path/to/processed_dataset
```

The processed dataset contains MRI and lesion masks in MNI/atlas space together with the transformation information required for native-space reconstruction.

A typical output structure is:

```text
processed_dataset/
└── mni_processed/
    └── sub-XXXX/
        ├── mri_mni.nii.gz
        ├── mask_mni.nii.gz
        ├── atlas_registration/
        ├── n4/
        ├── transforms/
        └── preprocessing.log
```

---

# 7. nnU-Net Integration

MedNeXt++ is integrated into **nnU-Net v2** through the custom trainer:

```text
training/nnUNetTrainer_MedNeXtPP.py
```

The trainer selects:

```python
MODEL_ID = "B"
```

and instantiates the network with:

```text
Kernel size        : 3
Deep supervision   : Enabled
```

The MedNeXt++ model implementation is kept separately from the trainer so that the architecture can be reused independently of nnU-Net.

---

# 8. Training

After atlas-centric preprocessing, the MNI-space images and labels should be organized into the standard nnU-Net dataset structure.

A typical nnU-Net dataset has:

```text
DatasetXXX/
├── imagesTr/
├── labelsTr/
├── imagesTs/        # optional
└── dataset.json
```

The standard nnU-Net v2 experiment-planning and preprocessing workflow can then be used.

Example:

```bash
nnUNetv2_plan_and_preprocess \
    -d 701 \
    -c 3d_fullres \
    --verify_dataset_integrity
```

Example training command:

```bash
nnUNetv2_train \
    701 \
    3d_fullres \
    0 \
    -tr nnUNetTrainer_MedNeXtPP
```

Here:

- `701` is the dataset ID used in the reported experiment
- `3d_fullres` is the full-resolution nnU-Net configuration
- `0` specifies fold 0
- `nnUNetTrainer_MedNeXtPP` selects the proposed trainer

---

# 9. Inference

Inference operates on an already preprocessed MNI-space MRI using a trained MedNeXt++ checkpoint.

Implementation:

```text
inference/inference.py
```

The inference pipeline is:

```text
MNI-space MRI
      │
      ▼
Trained MedNeXt++
      │
      ▼
MNI-space Prediction
      │
      ▼
Inverse Affine Transformation
      │
      ▼
Native-space Prediction
      │
      ▼
Three-plane Visualization
```

## 9.1 Running Inference

Example:

```bash
python inference/inference.py \
    --mni-mri /path/to/mri_mni.nii.gz \
    --model-results /path/to/nnUNet_results \
    --affine /path/to/affine_transform.mat \
    --native-mri /path/to/native_T1w.nii.gz \
    --output-dir /path/to/results
```

Optional arguments include:

```text
--dataset-id
--configuration
--trainer
--fold
--no-visualization
```

Default configuration:

```text
Dataset ID       : 701
Configuration    : 3d_fullres
Trainer          : nnUNetTrainer_MedNeXtPP
Fold             : 0
```

---

# 10. Native-Space Reconstruction

The trained model operates in MNI/atlas space.

For final output, the MNI-space segmentation is transformed back to the original patient/native space using the inverse of the affine transformation generated during preprocessing.

The segmentation uses nearest-neighbor interpolation during the inverse transformation.

This produces:

```text
prediction_native.nii.gz
```

which is aligned with the original native-space MRI geometry.

---

# 11. Visualization

The inference pipeline includes native-space visualization.

Three anatomical planes are generated:

- Axial
- Coronal
- Sagittal

Each view displays the native MRI together with the MedNeXt++ lesion prediction.

The output is saved as:

```text
prediction_visualization.png
```

Visualization can be disabled with:

```bash
--no-visualization
```

---

# 12. Output Structure

A successful inference run produces:

```text
results/
├── prediction_mni.nii.gz
├── prediction_native.nii.gz
└── prediction_visualization.png
```

### `prediction_mni.nii.gz`

The segmentation predicted by MedNeXt++ in MNI/atlas space.

### `prediction_native.nii.gz`

The segmentation transformed back into the original patient/native space.

### `prediction_visualization.png`

Three-plane visualization of the native MRI and predicted lesion.

---

# 13. Repository Structure

```text
MedNeXtPP-ISLES/
│
├── model/
│   ├── __init__.py
│   ├── blocks.py
│   ├── MedNextPP.py
│   └── create_mednextpp.py
│
├── preprocessing/
│   └── atlas_preprocessing.py
│
├── training/
│   └── nnUNetTrainer_MedNeXtPP.py
│
├── inference/
│   └── inference.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 14. Reproducibility

This repository provides the principal implementation components required to reproduce the proposed method:

1. Atlas-centric preprocessing
2. MedNeXt++ architecture
3. nnU-Net v2 training integration
4. MNI-space inference
5. Inverse spatial transformation
6. Native-space prediction
7. Visualization

The dataset and trained checkpoints are not distributed with this repository.

The formal experimental validation protocol, cross-validation strategy, and reported performance statistics are described in the accompanying paper.

---

# 15. Requirements

The main software dependencies are listed in:

```text
requirements.txt
```

The repository uses:

- PyTorch
- nnU-Net v2
- BrainLes preprocessing
- ANTsPy
- SimpleITK
- NumPy
- SciPy
- NiBabel
- scikit-image
- Matplotlib

---

# 16. Citation

If you use this implementation, please cite the associated publication:

```bibtex
@inproceedings{MEDNEXTPP_ISLES_2026,
    title     = {<FINAL PAPER TITLE>},
    author    = {<AUTHORS>},
    booktitle = {MICCAI SWITCH+ Workshop},
    year      = {2026}
}
```

The citation information will be updated with the final publication details.

---

# 17. License

This repository is intended for research and academic use.

Please also refer to the licenses and attribution requirements of the third-party frameworks and implementations used by this project, including nnU-Net, MedNeXt, BrainLes preprocessing, ANTs, and PyTorch.
