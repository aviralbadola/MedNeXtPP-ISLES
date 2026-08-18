"""
Atlas-centric preprocessing for the ATLAS R3.0 dataset.

Pipeline
--------
Native T1-weighted MRI
    -> Affine registration to MNI152 atlas
    -> N4 bias-field correction
    -> Percentile normalization
    -> MRI in MNI/atlas space

Native lesion mask
    -> Same affine transformation
    -> Nearest-neighbor interpolation
    -> Lesion mask in MNI/atlas space

The preprocessing implementation follows the configuration used in the
experimental pipeline and is exposed here as a reusable command-line script.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from brainles_preprocessing.preprocessor import AtlasCentricPreprocessor
from brainles_preprocessing.modality import CenterModality
from brainles_preprocessing.registration import ANTsRegistrator
from brainles_preprocessing.n4_bias_correction import SitkN4BiasCorrector
from brainles_preprocessing.normalization import PercentileNormalizer


# ============================================================================
# CONFIGURATION
# ============================================================================

REGISTRATION_PARAMS = {
    "type_of_transform": "Affine",
}

N4_MAX_ITERATIONS = 100

NORMALIZER_CONFIG = {
    "lower_percentile": 0,
    "upper_percentile": 100,
    "lower_limit": 0,
    "upper_limit": 1,
}


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def discover_subjects(
    dataset_root: Path,
    bad_subjects: set[str],
) -> list[Path]:
    """
    Find ATLAS subject anatomical directories.

    Expected layout:
        dataset_root/
            .../
                sub-XXXX/
                    ses-1/
                        anat/
    """
    subjects = []

    for anat_dir in sorted(dataset_root.glob("*/sub-*/ses-1/anat")):
        subject_id = anat_dir.parent.parent.name

        if subject_id in bad_subjects:
            continue

        subjects.append(anat_dir)

    return subjects


def find_input_files(subject_dir: Path) -> tuple[Path, Path]:
    """
    Find the native MRI and lesion mask for one subject.
    """
    mri_matches = list(subject_dir.glob("*desc-brain_T1w.nii.gz"))
    mask_matches = list(subject_dir.glob("*lesion*.nii.gz"))

    if not mri_matches:
        raise FileNotFoundError(
            f"No brain-extracted T1w MRI found in: {subject_dir}"
        )

    if not mask_matches:
        raise FileNotFoundError(
            f"No lesion mask found in: {subject_dir}"
        )

    return mri_matches[0], mask_matches[0]


# ============================================================================
# SINGLE SUBJECT PREPROCESSING
# ============================================================================

def preprocess_subject(
    subject_dir: Path,
    atlas_path: Path,
    output_root: Path,
    temp_root: Path,
    registrator: ANTsRegistrator,
    n4_corrector: SitkN4BiasCorrector,
    normalizer: PercentileNormalizer,
) -> tuple[str, str]:
    """
    Preprocess one subject.

    Returns
    -------
    (status, subject_id)

    status is one of:
        SUCCESS
        SKIPPED
    """

    subject_id = subject_dir.parent.parent.name

    # ----------------------------------------------------------------------
    # Input files
    # ----------------------------------------------------------------------

    mri_path, mask_path = find_input_files(subject_dir)

    # ----------------------------------------------------------------------
    # Output directories
    # ----------------------------------------------------------------------

    subject_output_dir = (
        output_root / "mni_processed" / subject_id
    )
    subject_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_temp_dir = temp_root / subject_id
    subject_temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_mri = subject_output_dir / "mri_mni.nii.gz"
    output_mask = subject_output_dir / "mask_mni.nii.gz"

    # ----------------------------------------------------------------------
    # Skip already-processed subjects
    # ----------------------------------------------------------------------

    if output_mri.exists() and output_mask.exists():
        shutil.rmtree(
            subject_temp_dir,
            ignore_errors=True,
        )
        return "SKIPPED", subject_id

    try:

        # ==================================================================
        # BrainLes center modality
        # ==================================================================

        center_modality = CenterModality(
            modality_name="T1w",
            input_path=mri_path,
            normalizer=normalizer,
            raw_skull_output_path=output_mri,
            atlas_correction=False,
            n4_bias_correction=True,
        )

        # ==================================================================
        # BrainLes atlas-centric preprocessor
        # ==================================================================

        preprocessor = AtlasCentricPreprocessor(
            center_modality=center_modality,
            moving_modalities=[],
            registrator=registrator,
            brain_extractor=None,
            defacer=None,
            atlas_image_path=atlas_path,
            n4_bias_corrector=n4_corrector,
            temp_folder=subject_temp_dir,
            use_gpu=False,
        )

        # ==================================================================
        # MRI preprocessing
        # ==================================================================

        preprocessor.run(
            save_dir_atlas_registration=(
                subject_output_dir / "atlas_registration"
            ),
            save_dir_n4_bias_correction=(
                subject_output_dir / "n4"
            ),
            save_dir_transformations=(
                subject_output_dir / "transforms"
            ),
            log_file=(
                subject_output_dir / "preprocessing.log"
            ),
        )

        if not output_mri.exists():
            raise RuntimeError(
                f"{subject_id}: MRI preprocessing failed."
            )

        # ==================================================================
        # Locate the registration transform
        # ==================================================================

        transform_files = list(
            (subject_output_dir / "transforms").rglob("*.mat")
        )

        if len(transform_files) != 1:
            raise RuntimeError(
                f"{subject_id}: Expected exactly one transform, "
                f"found {len(transform_files)}."
            )

        transform_file = transform_files[0]

        # ==================================================================
        # Transform lesion mask to atlas space
        # ==================================================================

        registrator.transform(
            fixed_image_path=atlas_path,
            moving_image_path=mask_path,
            transformed_image_path=output_mask,
            matrix_path=transform_file,
            log_file_path=(
                subject_output_dir / "mask_transform.log"
            ),
            interpolator="nearestNeighbor",
        )

        if not output_mask.exists():
            raise RuntimeError(
                f"{subject_id}: Mask transformation failed."
            )

        return "SUCCESS", subject_id

    finally:
        # Temporary registration/intermediate files are not needed after
        # preprocessing, but the final transform and logs are retained.
        shutil.rmtree(
            subject_temp_dir,
            ignore_errors=True,
        )


# ============================================================================
# BATCH PREPROCESSING
# ============================================================================

def run_preprocessing(
    dataset_root: Path,
    atlas_path: Path,
    output_root: Path,
    bad_subjects: set[str],
) -> None:
    """
    Run preprocessing for the complete dataset.
    """

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_root = output_root / "temp"
    temp_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Build BrainLes components
    # ----------------------------------------------------------------------

    registrator = ANTsRegistrator(
        registration_params=REGISTRATION_PARAMS
    )

    n4_corrector = SitkN4BiasCorrector(
        n_max_iterations=N4_MAX_ITERATIONS
    )

    normalizer = PercentileNormalizer(
        **NORMALIZER_CONFIG
    )

    # ----------------------------------------------------------------------
    # Discover subjects
    # ----------------------------------------------------------------------

    subjects = discover_subjects(
        dataset_root=dataset_root,
        bad_subjects=bad_subjects,
    )

    print("=" * 70)
    print("ATLAS R3.0 ATLAS-CENTRIC PREPROCESSING")
    print("=" * 70)
    print(f"Dataset root      : {dataset_root}")
    print(f"Atlas template    : {atlas_path}")
    print(f"Output root       : {output_root}")
    print(f"Subjects found    : {len(subjects)}")
    print(f"Skipped subjects  : {len(bad_subjects)}")
    print("=" * 70)

    successful_subjects: list[str] = []
    skipped_subjects: list[str] = []
    failed_subjects: list[dict[str, str]] = []

    print("\nStarting preprocessing...\n")

    try:
        for subject_dir in tqdm(
            subjects,
            desc="Preprocessing subjects",
        ):
            try:
                status, subject_id = preprocess_subject(
                    subject_dir=subject_dir,
                    atlas_path=atlas_path,
                    output_root=output_root,
                    temp_root=temp_root,
                    registrator=registrator,
                    n4_corrector=n4_corrector,
                    normalizer=normalizer,
                )

                if status == "SUCCESS":
                    successful_subjects.append(subject_id)

                elif status == "SKIPPED":
                    skipped_subjects.append(subject_id)

            except Exception as exc:
                subject_id = subject_dir.parent.parent.name

                failed_subjects.append(
                    {
                        "subject_id": subject_id,
                        "error": str(exc),
                    }
                )

                print(
                    f"\nFailed: {subject_id}\n"
                    f"Reason: {exc}"
                )

    except KeyboardInterrupt:
        print("\n\nPreprocessing interrupted by user.")

    finally:

        # --------------------------------------------------------------
        # Failure log
        # --------------------------------------------------------------

        failed_csv = output_root / "failed_subjects.csv"

        pd.DataFrame(
            failed_subjects
        ).to_csv(
            failed_csv,
            index=False,
        )

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        print("\n")
        print("=" * 70)
        print("PREPROCESSING SUMMARY")
        print("=" * 70)

        print(f"Total subjects : {len(subjects)}")
        print(f"Successful     : {len(successful_subjects)}")
        print(f"Skipped        : {len(skipped_subjects)}")
        print(f"Failed         : {len(failed_subjects)}")

        print("\nProcessed dataset:")
        print(output_root / "mni_processed")

        print("\nFailure log:")
        print(failed_csv)

        if failed_subjects:
            print("\nFailed subjects:")
            for item in failed_subjects:
                print(f"  - {item['subject_id']}")

        print("=" * 70)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BrainLes-style atlas-centric preprocessing "
            "for the ATLAS R3.0 dataset."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing the raw ATLAS dataset.",
    )

    parser.add_argument(
        "--atlas",
        type=Path,
        required=True,
        help="Path to the MNI152 T1 1mm atlas image.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where the processed dataset will be written.",
    )

    parser.add_argument(
        "--skip-subject",
        action="append",
        default=[],
        help=(
            "Subject ID to skip. Can be provided multiple times."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    atlas_path = args.atlas.resolve()
    output_root = args.output_root.resolve()

    # These are the two subjects excluded in the original pipeline.
    #
    # They are kept as defaults for reproducibility, but the command-line
    # option allows the list to be extended without editing the source code.
    bad_subjects = {
        "sub-r032s013",
        "sub-r032s018",
    }

    bad_subjects.update(args.skip_subject)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}"
        )

    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Atlas file does not exist: {atlas_path}"
        )

    run_preprocessing(
        dataset_root=dataset_root,
        atlas_path=atlas_path,
        output_root=output_root,
        bad_subjects=bad_subjects,
    )


if __name__ == "__main__":
    main()
