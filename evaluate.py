from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from surface_distance import compute_surface_dice_at_tolerance, compute_surface_distances
from tqdm.auto import tqdm


# taken from https://github.com/wasserth/TotalSegmentator/blob/master/resources/evaluate.py
def dice_score(y_true, y_pred):
    intersect = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    f1 = (2 * intersect) / (denominator + 1e-6)
    return f1


def calc_metrics(gt_all, pred_all, classes):
    r = {}
    for idx in range(1, classes + 1):
        roi_name = f"roi_{idx}"
        gt = gt_all == idx
        pred = pred_all == idx

        if gt.max() > 0 and pred.max() == 0:
            r[f"dice-{roi_name}"] = 0
            r[f"surface_dice_3-{roi_name}"] = 0
        elif gt.max() > 0:
            r[f"dice-{roi_name}"] = dice_score(gt, pred)
            sd = compute_surface_distances(gt, pred, [1.5, 1.5, 1.5])
            r[f"surface_dice_3-{roi_name}"] = compute_surface_dice_at_tolerance(sd, 3.0)
        # gt.max() == 0 which means we can not calculate any score because roi not in the image
        else:
            r[f"dice-{roi_name}"] = np.NaN
            r[f"surface_dice_3-{roi_name}"] = np.NaN
    return r


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("gt_path", type=str)
    parser.add_argument("pred_path", type=str)

    args = parser.parse_args()

    gt_path = Path(args.gt_path)
    predictions_path = Path(args.pred_path)
    classes = 13
    class_dices = {}
    all_metrics = []
    for file in tqdm(gt_path.glob("*.nii.gz"), total=len(list(gt_path.glob("*.nii.gz")))):
        gt = nib.load(file).get_fdata()
        pred = nib.load(predictions_path / file.name).get_fdata()
        metrics = calc_metrics(gt, pred, classes)
        all_metrics.append(metrics)

    res = pd.DataFrame(all_metrics)
    for metric in ["dice", "surface_dice_3"]:
        res_all_rois = []
        for idx in range(1, classes + 1):
            roi_name = f"roi_{idx}"
            row_wo_nan = res[f"{metric}-{roi_name}"].dropna()
            res_all_rois.append(row_wo_nan.mean())
            print(f"{roi_name} {metric}: {row_wo_nan.mean():.3f}")
        print(f"{metric}: {np.array(res_all_rois).mean():.3f}")
