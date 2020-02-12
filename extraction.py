import cv2 as cv
import numpy as np
import SIFT as sft
import VideoCapture_features as vc


#Lucas Kanade Params
lk_params = dict(winSize = (15,15),
                 maxLevel = 2,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def sift_kp_vector(sift_kp):
    vet = []
    for kp in sift_kp:
        point = [kp.pt[0],kp.pt[1]]
        vet.append(point)

    vet = np.array(vet)
    vet = np.float32(vet[:, np.newaxis, :])

    return vet


def compare_frames_opf(video_name):
    number_of_frames = vc.count_frames(video_name)
    for i in range(number_of_frames-2):
        actual_frame = vc.capture_frame(video_name, i+1)
        old_frame = vc.capture_frame(video_name, i)
        actual_gray_frame = sft.to_gray(actual_frame)
        old_gray_frame = sft.to_gray(old_frame)

        frame_kp, frame_desc = sft.sift_features(old_gray_frame)

        old_frame_points = sift_kp_vector(frame_kp)

        new_frame_points, status, error = cv.calcOpticalFlowPyrLK(old_gray_frame, actual_gray_frame, old_frame_points, None, **lk_params)

        print(new_frame_points)





