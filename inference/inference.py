"""
MedNeXt++ inference pipeline.

Pipeline
--------
MNI-space MRI
    -> trained MedNeXt++ checkpoint
    -> probability prediction
    -> probability thresholding
    -> MNI-space binary segmentation
    -> inverse affine transformation
    -> native-space segmentation
    -> optional visualization
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
import shutil

import ants
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# =============================================================================
# INVERSE TRANSFORM
# =============================================================================

def inverse_transform(
    moving: ants.ANTsImage,
    reference: ants.ANTsImage,
    affine: Path,
) -> ants.ANTsImage:
    """
    Transform a segmentation from MNI space back to native space.
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
    Create axial, coronal and sagittal views of the
    native-space MedNeXt++ prediction.
    """

    raw = native_mri.numpy()
    prediction = native_prediction.numpy().astype(bool)

    # -------------------------------------------------------------------------
    # Select slices containing the largest predicted lesion
    # -------------------------------------------------------------------------

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
            pred = np.rot90(
                prediction[:, idx, :]
            )

        else:

            mri = np.rot90(raw[idx, :, :])
            pred = np.rot90(
                prediction[idx, :, :]
            )

        # ---------------------------------------------------------------------
        # Crop around brain
        # ---------------------------------------------------------------------

        mri, row_slice, col_slice = crop_to_brain(mri)

        pred = pred[row_slice, col_slice]

        # ---------------------------------------------------------------------
        # MRI
        # ---------------------------------------------------------------------

        axes[row, 0].imshow(
            mri,
            cmap="gray",
            interpolation="nearest",
        )

        axes[row, 0].set_title(
            f"{plane} - MRI"
        )

        # ---------------------------------------------------------------------
        # Prediction overlay
        # ---------------------------------------------------------------------

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
            cmap=ListedColormap(
                ["deepskyblue"]
            ),
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
# MNI-SPACE MODEL INFERENCE
# =============================================================================

def run_model_inference(
    mni_mri: Path,
    model_results: Path,
    dataset_id: int,
    configuration: str,
    trainer: str,
    fold: int,
    threshold: float = 0.35,
) -> Path:
    """
    Run nnU-Net inference using the trained MedNeXt++ model.

    The model produces probability maps, after which a custom lesion
    probability threshold is applied to generate a binary MNI-space mask.
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"

        input_dir.mkdir()
        output_dir.mkdir()

        # ---------------------------------------------------------------------
        # nnU-Net expected input
        # ---------------------------------------------------------------------

        input_case = input_dir / "case_0000.nii.gz"

        shutil.copy2(
            mni_mri,
            input_case,
        )

        # ---------------------------------------------------------------------
        # Run nnU-Net inference
        # ---------------------------------------------------------------------

        command = [
            "nnUNetv2_predict",

            "-i",
            str(input_dir),

            "-o",
            str(output_dir),

            "-d",
            str(dataset_id),

            "-c",
            configuration,

            "-tr",
            trainer,

            "-f",
            str(fold),

            "--save_probabilities",
        ]

        env = os.environ.copy()

        if model_results is not None:
            env["nnUNet_results"] = str(model_results)

        print("\nRunning MedNeXt++ inference...")
        print(" ".join(command))

        subprocess.run(
            command,
            check=True,
            env=env,
        )

        # ---------------------------------------------------------------------
        # Load saved probability map
        # ---------------------------------------------------------------------

        probability_file = output_dir / "case.npz"

        if not probability_file.exists():
            raise RuntimeError(
                "nnU-Net probability file was not found:\n"
                f"{probability_file}"
            )

        probability_data = np.load(
            probability_file
        )

        if "probabilities" not in probability_data:
            raise RuntimeError(
                "The .npz file does not contain a 'probabilities' array."
            )

        probabilities = probability_data["probabilities"]

        # ---------------------------------------------------------------------
        # Binary lesion segmentation
        #
        # Channel 1 = lesion probability for binary segmentation
        # ---------------------------------------------------------------------

        if probabilities.ndim < 2:
            raise RuntimeError(
                f"Unexpected probability shape: {probabilities.shape}"
            )

        if probabilities.shape[0] < 2:
            raise RuntimeError(
                "Expected a background channel and a lesion channel."
            )

        lesion_probability = probabilities[1]

        prediction_binary = (
            lesion_probability >= threshold
        ).astype(np.uint8)

        # ---------------------------------------------------------------------
        # Use MNI MRI geometry for output mask
        # ---------------------------------------------------------------------

        mni_reference = ants.image_read(
            str(mni_mri)
        )

        prediction_mni_ants = ants.from_numpy(
            prediction_binary,
            origin=mni_reference.origin,
            spacing=mni_reference.spacing,
            direction=mni_reference.direction,
        )

        final_prediction = (
            Path.cwd() / "prediction_mni.nii.gz"
        )

        ants.image_write(
            prediction_mni_ants,
            str(final_prediction),
        )

        print(
            f"\nApplied probability threshold: {threshold:.2f}"
        )

        print(
            "\nMNI-space prediction saved to:"
        )
        print(final_prediction)

    return final_prediction


# =============================================================================
# COMPLETE INFERENCE PIPELINE
# =============================================================================

def run_inference(
    mni_mri_path: Path,
    model_results: Path,
    affine_path: Path,
    native_mri_path: Path,
    output_dir: Path,
    dataset_id: int = 701,
    configuration: str = "3d_fullres",
    trainer: str = "nnUNetTrainer_MedNeXtPP",
    fold: int = 0,
    threshold: float = 0.35,
    visualize: bool = True,
) -> Path:
    """
    Complete inference pipeline:

        MNI MRI
            ↓
        MedNeXt++
            ↓
        probability map
            ↓
        threshold = 0.35
            ↓
        MNI binary prediction
            ↓
        inverse affine
            ↓
        native prediction
            ↓
        visualization
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load native MRI
    # -------------------------------------------------------------------------

    native_mri = ants.image_read(
        str(native_mri_path)
    )

    # -------------------------------------------------------------------------
    # Model inference in MNI space
    # -------------------------------------------------------------------------

    mni_prediction = run_model_inference(
        mni_mri=mni_mri_path,
        model_results=model_results,
        dataset_id=dataset_id,
        configuration=configuration,
        trainer=trainer,
        fold=fold,
        threshold=threshold,
    )

    # -------------------------------------------------------------------------
    # Move MNI prediction to output directory
    # -------------------------------------------------------------------------

    mni_prediction_output = (
        output_dir /
        "prediction_mni.nii.gz"
    )

    shutil.copy2(
        mni_prediction,
        mni_prediction_output,
    )

    # -------------------------------------------------------------------------
    # Load MNI prediction
    # -------------------------------------------------------------------------

    prediction_mni = ants.image_read(
        str(mni_prediction_output)
    )

    # -------------------------------------------------------------------------
    # Inverse transform: MNI -> native
    # -------------------------------------------------------------------------

    print(
        "\nApplying inverse affine transformation..."
    )

    prediction_native = inverse_transform(
        moving=prediction_mni,
        reference=native_mri,
        affine=affine_path,
    )

    native_prediction_path = (
        output_dir /
        "prediction_native.nii.gz"
    )

    ants.image_write(
        prediction_native,
        str(native_prediction_path),
    )

    print(
        "\nNative prediction saved to:"
    )

    print(
        native_prediction_path
    )

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

        print(
            "\nVisualization saved to:"
        )

        print(
            visualization_path
        )

    return native_prediction_path


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run MedNeXt++ inference from MNI space "
            "to native space."
        )
    )

    parser.add_argument(
        "--mni-mri",
        type=Path,
        required=True,
        help="Preprocessed MNI-space MRI.",
    )

    parser.add_argument(
        "--model-results",
        type=Path,
        required=True,
        help=(
            "nnUNet_results directory containing "
            "the trained MedNeXt++ model."
        ),
    )

    parser.add_argument(
        "--affine",
        type=Path,
        required=True,
        help=(
            "Native-to-MNI affine transform generated "
            "during preprocessing."
        ),
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
        help="Directory for prediction outputs.",
    )

    parser.add_argument(
        "--dataset-id",
        type=int,
        default=701,
        help="nnU-Net dataset ID.",
    )

    parser.add_argument(
        "--configuration",
        default="3d_fullres",
        help="nnU-Net configuration.",
    )

    parser.add_argument(
        "--trainer",
        default="nnUNetTrainer_MedNeXtPP",
        help="Custom nnU-Net trainer name.",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Cross-validation fold.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Probability threshold for lesion segmentation.",
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Do not generate the visualization.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    args = parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    required = {
        "MNI MRI": args.mni_mri,
        "model results": args.model_results,
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
        model_results=args.model_results,
        affine_path=args.affine,
        native_mri_path=args.native_mri,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        trainer=args.trainer,
        fold=args.fold,
        threshold=args.threshold,
        visualize=not args.no_visualization,
    )


if __name__ == "__main__":
    main()
