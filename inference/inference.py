"""
MedNeXt++ inference pipeline.

Pipeline
--------
MNI-space MRI
    ↓
Z-score normalization
    ↓
MedNeXt++ Base (B)
    ↓
Sliding-window inference
    ↓
Probability map
    ↓
Threshold = 0.35
    ↓
Binary MNI segmentation
    ↓
Inverse affine transform
    ↓
Native-space segmentation
    ↓
Optional visualization
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ants
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunet_mednext.network_architecture.mednextpp.create_mednextpp import (
    create_mednextpp,
)


# =============================================================================
# DEFAULT SETTINGS
# =============================================================================

DEFAULT_THRESHOLD = 0.35
DEFAULT_TILE_STEP_SIZE = 0.25
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_FOLD = 0


# =============================================================================
# INVERSE TRANSFORM
# =============================================================================

def inverse_transform(
    moving: ants.ANTsImage,
    reference: ants.ANTsImage,
    affine: Path,
) -> ants.ANTsImage:
    """
    Transform a binary segmentation from MNI space back to native space.
    """

    if not affine.exists():
        raise FileNotFoundError(
            f"Affine transform not found:\n{affine}"
        )

    return ants.apply_transforms(
        fixed=reference,
        moving=moving,
        transformlist=[str(affine)],
        whichtoinvert=[True],
        interpolator="nearestNeighbor",
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def crop_to_brain(
    image: np.ndarray,
    margin: int = 5,
):
    """
    Crop a 2D MRI slice around the non-zero brain region.
    """

    if np.max(image) == 0:
        return image, slice(None), slice(None)

    mask = image > (0.02 * np.max(image))

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return image, slice(None), slice(None)

    r0 = max(rows[0] - margin, 0)
    r1 = min(rows[-1] + margin + 1, image.shape[0])

    c0 = max(cols[0] - margin, 0)
    c1 = min(cols[-1] + margin + 1, image.shape[1])

    return (
        image[r0:r1, c0:c1],
        slice(r0, r1),
        slice(c0, c1),
    )


def visualize_prediction(
    native_mri: ants.ANTsImage,
    native_prediction: ants.ANTsImage,
    save_path: Path,
) -> None:
    """
    Create axial, coronal and sagittal views of the native-space prediction.
    """

    raw = native_mri.numpy()
    prediction = native_prediction.numpy().astype(bool)

    if np.any(prediction):

        axial_idx = np.argmax(
            prediction.sum(axis=(0, 1))
        )

        coronal_idx = np.argmax(
            prediction.sum(axis=(0, 2))
        )

        sagittal_idx = np.argmax(
            prediction.sum(axis=(1, 2))
        )

    else:

        axial_idx = raw.shape[2] // 2
        coronal_idx = raw.shape[1] // 2
        sagittal_idx = raw.shape[0] // 2

    planes = [
        ("Axial", axial_idx),
        ("Coronal", coronal_idx),
        ("Sagittal", sagittal_idx),
    ]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10, 12),
    )

    for row, (plane, idx) in enumerate(planes):

        if plane == "Axial":
            mri = raw[:, :, idx]
            pred = prediction[:, :, idx]

        elif plane == "Coronal":
            mri = np.rot90(raw[:, idx, :])
            pred = np.rot90(prediction[:, idx, :])

        else:
            mri = np.rot90(raw[idx, :, :])
            pred = np.rot90(prediction[idx, :, :])

        mri, row_slice, col_slice = crop_to_brain(mri)
        pred = pred[row_slice, col_slice]

        axes[row, 0].imshow(
            mri,
            cmap="gray",
            interpolation="nearest",
        )
        axes[row, 0].set_title(f"{plane} - MRI")

        axes[row, 1].imshow(
            mri,
            cmap="gray",
            interpolation="nearest",
        )

        axes[row, 1].imshow(
            np.ma.masked_where(
                pred == 0,
                pred,
            ),
            cmap=ListedColormap(["deepskyblue"]),
            alpha=0.55,
            interpolation="nearest",
        )

        axes[row, 1].set_title(
            f"{plane} - MedNeXt++"
        )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.legend(
        handles=[
            Patch(
                color="deepskyblue",
                label="MedNeXt++ prediction",
            )
        ],
        loc="lower center",
        frameon=False,
    )

    plt.tight_layout(
        rect=(0, 0.03, 1, 1)
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def initialize_model(
    checkpoint_path: Path,
    dataset_json_path: Path,
    plans_json_path: Path,
    configuration_name: str = DEFAULT_CONFIGURATION,
    tile_step_size: float = DEFAULT_TILE_STEP_SIZE,
) -> dict:
    """
    Initialize the exact MedNeXt++ Base (B) model and nnU-Net predictor.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    if not dataset_json_path.exists():
        raise FileNotFoundError(
            f"dataset.json not found:\n{dataset_json_path}"
        )

    if not plans_json_path.exists():
        raise FileNotFoundError(
            f"plans.json not found:\n{plans_json_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("MEDNEXT++ INITIALIZATION")
    print("=" * 70)
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # -------------------------------------------------------------------------
    # Exact MedNeXt++ model used in Docker
    # -------------------------------------------------------------------------

    model = create_mednextpp(
        num_input_channels=1,
        num_classes=2,
        model_id="B",
        kernel_size=3,
        deep_supervision=True,
    )

    missing, unexpected = model.load_state_dict(
        checkpoint["network_weights"],
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch detected.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # Load nnU-Net metadata
    # -------------------------------------------------------------------------

    dataset_json = load_json(
        str(dataset_json_path)
    )

    plans = load_json(
        str(plans_json_path)
    )

    plans_manager = PlansManager(plans)

    configuration = plans_manager.get_configuration(
        configuration_name
    )

    # -------------------------------------------------------------------------
    # Exact predictor settings from Docker
    # -------------------------------------------------------------------------

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(
            device.type == "cuda"
        ),
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.network = model
    predictor.device = device
    predictor.configuration_manager = configuration
    predictor.plans_manager = plans_manager
    predictor.dataset_json = dataset_json
    predictor.label_manager = (
        plans_manager.get_label_manager(
            dataset_json
        )
    )

    if "inference_allowed_mirroring_axes" in checkpoint:
        predictor.allowed_mirroring_axes = (
            checkpoint["inference_allowed_mirroring_axes"]
        )

    print("Model: MedNeXt++ Base (B)")
    print(f"Tile step size: {tile_step_size}")
    print("Gaussian weighting: True")
    print("Mirroring: True")
    print("=" * 70)

    return {
        "model": model,
        "device": device,
        "predictor": predictor,
    }


# =============================================================================
# MEDNEXT++ MNI INFERENCE
# =============================================================================

def run_model_inference(
    mni_mri_path: Path,
    predictor: nnUNetPredictor,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
) -> Path:
    """
    Run MedNeXt++ inference directly on an MNI-space MRI.
    """

    if not mni_mri_path.exists():
        raise FileNotFoundError(
            f"MNI MRI not found:\n{mni_mri_path}"
        )

    mni_sitk = ants.image_read(
        str(mni_mri_path)
    )

    mni_array = (
        mni_sitk.numpy()
        .astype(np.float32)
    )

    # -------------------------------------------------------------------------
    # Same z-score normalization used in Docker inference
    # -------------------------------------------------------------------------

    mean = mni_array.mean()
    std = mni_array.std()

    mni_array = (
        mni_array - mean
    ) / (std + 1e-8)

    # -------------------------------------------------------------------------
    # Prepare predictor input: [C, Z, Y, X]
    # -------------------------------------------------------------------------

    image = (
        torch.from_numpy(mni_array)
        .float()
        .unsqueeze(0)
    )

    print("\nRunning MedNeXt++ sliding-window inference...")

    with torch.no_grad():
        logits = predictor.predict_sliding_window_return_logits(
            image
        )

    if isinstance(logits, torch.Tensor):
        logits = (
            logits
            .detach()
            .cpu()
            .numpy()
        )

    logits = np.asarray(
        logits,
        dtype=np.float32,
    )

    if logits.ndim == 5:
        logits = logits[0]

    if logits.ndim != 4:
        raise RuntimeError(
            f"Unexpected logits shape: {logits.shape}"
        )

    if logits.shape[0] != 2:
        raise RuntimeError(
            f"Expected 2 output classes, got {logits.shape[0]}"
        )

    # -------------------------------------------------------------------------
    # Logits -> probabilities
    # -------------------------------------------------------------------------

    probabilities = torch.softmax(
        torch.from_numpy(logits),
        dim=0,
    ).numpy()

    lesion_probability = (
        probabilities[1]
        .astype(np.float32)
    )

    # -------------------------------------------------------------------------
    # Probability -> binary mask
    # -------------------------------------------------------------------------

    lesion_mask = (
        lesion_probability >= threshold
    ).astype(np.uint8)

    print(
        f"Probability threshold: {threshold:.2f}"
    )

    print(
        "MNI lesion voxels:",
        int(lesion_mask.sum()),
    )

    # -------------------------------------------------------------------------
    # Save MNI prediction
    # -------------------------------------------------------------------------

    mni_prediction = ants.from_numpy(
        lesion_mask,
        origin=mni_sitk.origin,
        spacing=mni_sitk.spacing,
        direction=mni_sitk.direction,
    )

    output_path = (
        Path.cwd() /
        "prediction_mni.nii.gz"
    )

    ants.image_write(
        mni_prediction,
        str(output_path),
    )

    return output_path


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def run_inference(
    mni_mri_path: Path,
    checkpoint_path: Path,
    dataset_json_path: Path,
    plans_json_path: Path,
    affine_path: Path,
    native_mri_path: Path,
    output_dir: Path,
    configuration: str = DEFAULT_CONFIGURATION,
    fold: int = DEFAULT_FOLD,
    threshold: float = DEFAULT_THRESHOLD,
    tile_step_size: float = DEFAULT_TILE_STEP_SIZE,
    visualize: bool = True,
) -> Path:

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if not 0.0 < tile_step_size <= 1.0:
        raise ValueError(
            "Tile step size must be > 0 and <= 1."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Initialize model
    # -------------------------------------------------------------------------

    model_info = initialize_model(
        checkpoint_path=checkpoint_path,
        dataset_json_path=dataset_json_path,
        plans_json_path=plans_json_path,
        configuration_name=configuration,
        tile_step_size=tile_step_size,
    )

    predictor = model_info["predictor"]
    device = model_info["device"]

    # -------------------------------------------------------------------------
    # Load native MRI
    # -------------------------------------------------------------------------

    native_mri = ants.image_read(
        str(native_mri_path)
    )

    # -------------------------------------------------------------------------
    # MNI inference
    # -------------------------------------------------------------------------

    mni_prediction = run_model_inference(
        mni_mri_path=mni_mri_path,
        predictor=predictor,
        device=device,
        threshold=threshold,
    )

    mni_output = (
        output_dir /
        "prediction_mni.nii.gz"
    )

    mni_prediction_img = ants.image_read(
        str(mni_prediction)
    )

    ants.image_write(
        mni_prediction_img,
        str(mni_output),
    )

    # -------------------------------------------------------------------------
    # MNI -> native
    # -------------------------------------------------------------------------

    print("\nApplying inverse affine transformation...")

    prediction_native = inverse_transform(
        moving=mni_prediction_img,
        reference=native_mri,
        affine=affine_path,
    )

    native_output = (
        output_dir /
        "prediction_native.nii.gz"
    )

    ants.image_write(
        prediction_native,
        str(native_output),
    )

    print("\nNative prediction saved to:")
    print(native_output)

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    if visualize:

        visualization_path = (
            output_dir /
            "prediction_visualization.png"
        )

        visualize_prediction(
            native_mri=native_mri,
            native_prediction=prediction_native,
            save_path=visualization_path,
        )

        print("\nVisualization saved to:")
        print(visualization_path)

    return native_output


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run MedNeXt++ inference from MNI space "
            "to native patient space."
        )
    )

    parser.add_argument(
        "--mni-mri",
        type=Path,
        required=True,
        help="Preprocessed MNI-space MRI.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="MedNeXt++ checkpoint_best.pth.",
    )

    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
        help="nnU-Net dataset.json.",
    )

    parser.add_argument(
        "--plans-json",
        type=Path,
        required=True,
        help="nnU-Net plans.json.",
    )

    parser.add_argument(
        "--affine",
        type=Path,
        required=True,
        help="Native-to-MNI affine transform.",
    )

    parser.add_argument(
        "--native-mri",
        type=Path,
        required=True,
        help="Original native-space MRI.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inference_results"),
        help="Output directory.",
    )

    parser.add_argument(
        "--configuration",
        default=DEFAULT_CONFIGURATION,
        help="nnU-Net configuration.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=DEFAULT_FOLD,
        help="Fold number.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Lesion probability threshold.",
    )

    parser.add_argument(
        "--tile-step-size",
        type=float,
        default=DEFAULT_TILE_STEP_SIZE,
        help="Sliding-window tile step size.",
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable visualization.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    required = {
        "MNI MRI": args.mni_mri,
        "checkpoint": args.checkpoint,
        "dataset.json": args.dataset_json,
        "plans.json": args.plans_json,
        "affine transform": args.affine,
        "native MRI": args.native_mri,
    }

    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} does not exist:\n{path}"
            )

    run_inference(
        mni_mri_path=args.mni_mri,
        checkpoint_path=args.checkpoint,
        dataset_json_path=args.dataset_json,
        plans_json_path=args.plans_json,
        affine_path=args.affine,
        native_mri_path=args.native_mri,
        output_dir=args.output_dir,
        configuration=args.configuration,
        fold=args.fold,
        threshold=args.threshold,
        tile_step_size=args.tile_step_size,
        visualize=not args.no_visualization,
    )


if __name__ == "__main__":
    main()
