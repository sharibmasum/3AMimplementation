import os
import cv2
import numpy as np

def _load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    return (mask > 0).astype(np.uint8)

def _iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0  # both eempty perfect overlap
    return intersection / union

def evaluate_masks(pred_dir, gt_dir, success_threshold=0.5):
    ious = []
    positive_ious = []
    successful = 0
    total = 0

    for name in os.listdir(gt_dir):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        gt_path = os.path.join(gt_dir, name)
        pred_path = os.path.join(pred_dir, name)

        if not os.path.exists(pred_path):
            continue

        gt = _load_mask(gt_path)
        pred = _load_mask(pred_path)

        iou = _iou(pred, gt)
        ious.append(iou)
        total += 1

        if gt.sum() > 0:
            positive_ious.append(iou)

        if iou >= success_threshold:
            successful += 1

    return {
        "iou": float(np.mean(ious)) if ious else 0.0,
        "positive_iou": float(np.mean(positive_ious)) if positive_ious else 0.0,
        "successful_iou": successful / total if total > 0 else 0.0,
    }
