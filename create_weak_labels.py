from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator

label_files_order = [
    "spleen.nii.gz",  # label 1
    "kidney_right.nii.gz",  # label 2
    "kidney_left.nii.gz",  # label 3
    "gallbladder.nii.gz",  # label 4
    "esophagus.nii.gz",
    "liver.nii.gz",
    "stomach.nii.gz",
    "aorta.nii.gz",
    "inferior_vena_cava.nii.gz",
    "portal_vein_and_splenic_vein.nii.gz",
    "pancreas.nii.gz",
    "adrenal_gland_right.nii.gz",
    "adrenal_gland_left.nii.gz",
]
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("root_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    nii_files = Path(args.root_path).glob("**/*.nii.gz")
    labels_folder = Path(args.output_path) / "labels"
    labels_folder.mkdir(exist_ok=True, parents=True)

    for nii_file in nii_files:
        output_folder = Path(args.output_path) / (nii_file.stem[:-4])

        totalsegmentator(
            nii_file,
            output_folder,
            fast=True,
            roi_subset=[
                "spleen",
                "kidney_right",
                "kidney_left",
                "gallbladder",
                "liver",
                "stomach",
                "aorta",
                "inferior_vena_cava",
                "portal_vein_and_splenic_vein",
                "pancreas",
                "adrenal_gland_right",
                "adrenal_gland_left",
                "esophagus",
            ],
        )

        masks = []
        for i, f in enumerate(label_files_order, start=1):
            m = nib.load(output_folder / f).get_fdata()
            m = m * i
            masks.append(m)
        mask = np.sum(masks, axis=0)

        # Load the original NIfTI file to get the header and affine
        original_nii = nib.load(nii_file)
        new_nii = nib.Nifti1Image(mask.astype(np.int32), original_nii.affine, original_nii.header)
        new_nii.header.set_data_dtype(np.int32)
        new_nii.to_filename(labels_folder / f"{nii_file.name.replace('img','label')}")
