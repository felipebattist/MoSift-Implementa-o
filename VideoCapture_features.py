import cv2 as cv
import numpy as np

def capture_frame(video, frame):
    cap = cv.VideoCapture(video)
    cap.set(1, frame)
    ret, frame = cap.read()

    return frame

def count_frames(video):
    cap = cv.VideoCapture(video)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    return total


