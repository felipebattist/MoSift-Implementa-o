import cv2 as cv
import numpy as np
import SIFT as sft
import VideoCapture_features as vc


#Lucas Kanade Params
lk_params = dict(winSize = (15,15),
                 maxLevel = 2,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def sift_kp_2d(sift_kp):
    vet = []
    for kp in sift_kp:
        point = [kp.pt[0],kp.pt[1]]
        vet.append(point)

    vet = np.array(vet)
    vet = np.float32(vet[:, np.newaxis, :])

    return vet


def kp_moviment(sift_kp, sift_desc, old_points, new_points, distance_limit):
    for i in range(len(old_points)):
        new_sift_kp = []
        new_sift_desc = []
        selected_points = []
        distance_of_x = int(old_points[i].T[0]) - int(new_points[i].T[0])
        distance_of_y = int(old_points[i].T[1]) - int(new_points[i].T[1])
        if(distance_of_x < 0):
            distance_of_x = (distance_of_x * -1)
        if (distance_of_y < 0):
            distance_of_y = (distance_of_y * -1)

        if distance_of_x > distance_limit or distance_of_y > distance_limit:
            new_sift_kp.append(sift_kp[i])
            new_sift_desc.append(sift_kp[i])
            interest_point = [int(old_points[i].T[0]), int(old_points[i].T[1])]
            selected_points.append(interest_point)

    return new_sift_kp, new_sift_desc, selected_points




def compare_frames_opf(video_name, distance_limit):
    number_of_frames = vc.count_frames(video_name)
    for i in range(number_of_frames-2):
        actual_frame = vc.capture_frame(video_name, i+1)
        old_frame = vc.capture_frame(video_name, i)
        actual_gray_frame = sft.to_gray(actual_frame)
        old_gray_frame = sft.to_gray(old_frame)

        frame_kp, frame_desc = sft.sift_features(old_gray_frame)

        old_frame_points = sift_kp_2d(frame_kp)

        new_frame_points, status, error = cv.calcOpticalFlowPyrLK(old_gray_frame, actual_gray_frame, old_frame_points, None, **lk_params)

        frame_kp, frame_desc, frame_2d_kp = kp_moviment(frame_kp, frame_desc, old_frame_points, new_frame_points, distance_limit)

        print(old_frame_points)
        print('=========================================================================')
        print(frame_2d_kp)









