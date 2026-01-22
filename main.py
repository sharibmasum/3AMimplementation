import os
from sam2_model import SAM2
import cv2

def runFirstFrame():
    sam2 = SAM2()
    sam2.viewFirstFrame()


def runMaskFirstFrame(point):
    sam2 = SAM2()
    sam2.prepareModel()
    sam2.maskFirstFrame(point, show=True)


def run_segment_video(point):
    sam2 = SAM2()
    sam2.prepareModel()
    sam2.maskFirstFrame(point, show=False)
    sam2.segmentVideo()


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
    # getFrames('/Users/sharibmasum/PycharmProjects/3AMimplementation/videos/[video]')
    # runFirstFrame()
    # runMaskFirstFrame([[100, 460]])
    run_segment_video([[100, 460]])