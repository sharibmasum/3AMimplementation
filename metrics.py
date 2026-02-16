import os
import argparse
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

def evaluate_masks(pred_dir, gt_dir, success_threshold=0.5, strict_size=True, verbose=True):
    ious = []
    positive_ious = []
    successful = 0
    total = 0
    gt_files = sorted(
        name
        for name in os.listdir(gt_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    pred_files = set(
        name
        for name in os.listdir(pred_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    matched = [name for name in gt_files if name in pred_files]
    missing_pred = [name for name in gt_files if name not in pred_files]
    if verbose:
        print(f"Evaluating masks: gt={len(gt_files)} pred={len(pred_files)} matched={len(matched)}")
        if missing_pred:
            print(f"Missing predictions for {len(missing_pred)} gt masks.")

    for name in matched:
        gt_path = os.path.join(gt_dir, name)
        pred_path = os.path.join(pred_dir, name)
        gt = _load_mask(gt_path)
        pred = _load_mask(pred_path)
        if gt.shape != pred.shape:
            if strict_size:
                raise ValueError(
                    f"Mask size mismatch for {name}: pred={pred.shape} gt={gt.shape}"
                )
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

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


def compare_runs(pred_a_dir, pred_b_dir, gt_dir, success_threshold=0.5, strict_size=True):
    """Compare two prediction folders against the same GT."""
    a = evaluate_masks(
        pred_a_dir,
        gt_dir,
        success_threshold=success_threshold,
        strict_size=strict_size,
        verbose=False,
    )
    b = evaluate_masks(
        pred_b_dir,
        gt_dir,
        success_threshold=success_threshold,
        strict_size=strict_size,
        verbose=False,
    )
    return {
        "pred_a": a,
        "pred_b": b,
        "delta_iou": b["iou"] - a["iou"],
        "delta_positive_iou": b["positive_iou"] - a["positive_iou"],
        "delta_successful_iou": b["successful_iou"] - a["successful_iou"],
    }


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compute IoU for one run or compare two runs."
    )
    parser.add_argument("--pred", help="Prediction masks folder for single-run IoU.")
    parser.add_argument("--pred-a", help="Prediction masks folder for run A.")
    parser.add_argument("--pred-b", help="Prediction masks folder for run B.")
    parser.add_argument("--gt", required=True, help="Ground-truth masks folder.")
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--strict-size", action="store_true", default=True)
    parser.add_argument("--no-strict-size", action="store_false", dest="strict_size")
    return parser


def _print_results(title, results):
    print(title)
    print(f"  IoU: {results['iou']:.4f}")
    print(f"  Positive IoU: {results['positive_iou']:.4f}")
    print(f"  Successful IoU: {results['successful_iou']:.4f}")


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    if args.pred_a and args.pred_b:
        results = compare_runs(
            args.pred_a,
            args.pred_b,
            args.gt,
            success_threshold=args.success_threshold,
            strict_size=args.strict_size,
        )
        _print_results("Run A:", results["pred_a"])
        _print_results("Run B:", results["pred_b"])
        print("Deltas (B - A):")
        print(f"  IoU: {results['delta_iou']:.4f}")
        print(f"  Positive IoU: {results['delta_positive_iou']:.4f}")
        print(f"  Successful IoU: {results['delta_successful_iou']:.4f}")
    elif args.pred:
        results = evaluate_masks(
            args.pred,
            args.gt,
            success_threshold=args.success_threshold,
            strict_size=args.strict_size,
        )
        _print_results("Run:", results)
    else:
        raise SystemExit("Provide --pred or --pred-a and --pred-b.")
