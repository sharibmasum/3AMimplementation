import argparse
import os

from metrics import compare_runs
from sam2_model import SAM2


def _parse_point(text: str):
    for sep in (",", " "):
        if sep in text:
            parts = [p for p in text.replace(",", " ").split(" ") if p]
            if len(parts) == 2:
                return [[float(parts[0]), float(parts[1])]]
    raise ValueError("Point must be in 'x,y' or 'x y' format.")


def _clear_dir(path: str):
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(path, name))


def _run_one(point, pred_dir, output_video_path: str, disable_geometry: bool):
    if disable_geometry:
        os.environ["DISABLE_GEOMETRY"] = "1"
    else:
        os.environ.pop("DISABLE_GEOMETRY", None)
    sam2 = SAM2()
    sam2.prepareModel()
    geom_active = (
        getattr(sam2, "predictor", None) is not None
        and getattr(sam2.predictor, "feature_merger", None) is not None
    )
    print(f"DISABLE_GEOMETRY={os.getenv('DISABLE_GEOMETRY', '0')} | Geometry active: {geom_active}")
    sam2.maskFirstFrame(point, show=False)
    sam2.segmentVideo(
        export_annotations=False,
        pred_masks_dir=pred_dir,
        output_video_path=output_video_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline vs MUSt3R geometry fusion and compare IoU."
    )
    parser.add_argument(
        "--point",
        required=True,
        help="Click point as 'x,y' (pixel coordinates).",
    )
    parser.add_argument(
        "--gt",
        default=os.path.join(os.path.dirname(__file__), "videos", "gt_masks"),
        help="Ground-truth masks folder.",
    )
    parser.add_argument(
        "--pred-a",
        default=os.path.join(os.path.dirname(__file__), "videos", "pred_masks_baseline"),
        help="Output folder for baseline (geometry OFF).",
    )
    parser.add_argument(
        "--pred-b",
        default=os.path.join(os.path.dirname(__file__), "videos", "pred_masks_geom"),
        help="Output folder for geometry fusion (MUSt3R ON).",
    )
    args = parser.parse_args()

    point = _parse_point(args.point)
    _clear_dir(args.pred_a)
    _clear_dir(args.pred_b)

    print("Running baseline (geometry OFF)...")
    _run_one(
        point,
        args.pred_a,
        os.path.join(os.path.dirname(__file__), "videos", "output_segmented_baseline.mp4"),
        disable_geometry=True,
    )
    print("Running geometry fusion (MUSt3R ON)...")
    _run_one(
        point,
        args.pred_b,
        os.path.join(os.path.dirname(__file__), "videos", "output_segmented_geom.mp4"),
        disable_geometry=False,
    )

    results = compare_runs(args.pred_a, args.pred_b, args.gt)
    print("Baseline vs Geometry (IoU):")
    print(f"  Baseline IoU: {results['pred_a']['iou']:.4f}")
    print(f"  Geometry IoU: {results['pred_b']['iou']:.4f}")
    print(f"  Delta IoU: {results['delta_iou']:.4f}")
    print(f"  Delta Positive IoU: {results['delta_positive_iou']:.4f}")
    print(f"  Delta Successful IoU: {results['delta_successful_iou']:.4f}")


if __name__ == "__main__":
    main()

