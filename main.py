import os
from sam2_model import SAM2
from metrics import compare_runs
import cv2

def runFirstFrame():
    sam2 = SAM2()
    sam2.viewFirstFrame()

def runMaskFirstFrame(point):
    sam2 = SAM2()
    sam2.prepareModel()
    sam2.maskFirstFrame(point, show=True)


def run_segment_video(
    point,
    run_metrics=True,
    export_annotations=True,
    annotations_root=None,
    annotations_video_name="video1",
    pred_masks_dir=None,
):
    sam2 = SAM2()
    sam2.prepareModel()
    sam2.maskFirstFrame(point, show=False)
    sam2.segmentVideo(
        export_annotations=export_annotations,
        annotations_root=annotations_root,
        annotations_video_name=annotations_video_name,
        pred_masks_dir=pred_masks_dir,
    )
    if run_metrics:
        from metrics import evaluate_masks
        project_root = os.path.dirname(os.path.abspath(__file__))
        pred_dir = os.path.join(project_root, 'videos', 'pred_masks')
        gt_dir = os.path.join(project_root, 'videos', 'gt_masks')
        results = evaluate_masks(pred_dir, gt_dir)
        print("Whole Set:")
        print(f"  IoU: {results['iou']:.4f}")
        print(f"  Positive IoU: {results['positive_iou']:.4f}")
        print(f"  Successful IoU: {results['successful_iou']:.4f}")


def _clear_dir(path: str) -> None:
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


def run_segment_video_baseline_geom(
    point,
    run_metrics=True,
    pred_a_dir=None,
    pred_b_dir=None,
):
    project_root = os.path.dirname(os.path.abspath(__file__))
    pred_a_dir = pred_a_dir or os.path.join(project_root, "videos", "pred_masks_baseline")
    pred_b_dir = pred_b_dir or os.path.join(project_root, "videos", "pred_masks_geom")
    _clear_dir(pred_a_dir)
    _clear_dir(pred_b_dir)

    print("Running baseline (geometry OFF)...")
    _run_one(
        point,
        pred_a_dir,
        os.path.join(project_root, "videos", "output_segmented_baseline.mp4"),
        disable_geometry=True,
    )
    print("Running geometry fusion (MUSt3R ON)...")
    _run_one(
        point,
        pred_b_dir,
        os.path.join(project_root, "videos", "output_segmented_geom.mp4"),
        disable_geometry=False,
    )

    if run_metrics:
        gt_dir = os.path.join(project_root, "videos", "gt_masks")
        results = compare_runs(pred_a_dir, pred_b_dir, gt_dir)
        print("Baseline vs Geometry (IoU):")
        print(f"  Baseline IoU: {results['pred_a']['iou']:.4f}")
        print(f"  Geometry IoU: {results['pred_b']['iou']:.4f}")
        print(f"  Delta IoU: {results['delta_iou']:.4f}")
        print(f"  Delta Positive IoU: {results['delta_positive_iou']:.4f}")
        print(f"  Delta Successful IoU: {results['delta_successful_iou']:.4f}")


def extractFrames(videoPath, outputFolder, skip_frames): # extracting the frames, helper
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    capture = cv2.VideoCapture(videoPath)

    if not capture.isOpened():
        print("Cant open video file")
        return

    frameCount = 0
    savedCount = 0

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if frameCount % (skip_frames + 1) == 0:
            frame = cv2.resize(frame, (640, 480))
            fileName = f"{savedCount:04d}.jpg"
            filePath = os.path.join(outputFolder, fileName)
            cv2.imwrite(filePath, frame)
            print(f"Saved {fileName}")
            savedCount += 1
        frameCount += 1

    capture.release()
    print("done frame extraction")

def getFrames(videoFile): # helper function
    where = os.path.join(os.getcwd(), 'videos')
    videoPath = os.path.join(where, videoFile)
    outputFolder = os.path.join(where, 'frames')
    skip_frames = 5 # adjufst this later
    extractFrames(videoPath, outputFolder, skip_frames)


if __name__ == '__main__':
    # getFrames('/Users/sharibmasum/PycharmProjects/3AMimplementation/IMG_9746.mp4')
    # runFirstFrame()
    # runMaskFirstFrame([[180, 296]])
    run_segment_video_baseline_geom([[180, 296]])