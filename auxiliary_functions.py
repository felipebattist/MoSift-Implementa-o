import cv2 as cv
import numpy as np
import SIFT as sft
import VideoCapture_features as vc
import math

def gen_neighbors(x, y):
    n1 = [[x - 8, y + 8], [x - 7, y + 8], [x - 6, y + 8], [x - 5, y + 8],
          [x - 8, y + 7], [x - 7, y + 7], [x - 6, y + 7], [x - 5, y + 7],
          [x - 8, y + 6], [x - 7, y + 6], [x - 6, y + 6], [x - 5, y + 6],
          [x - 8, y + 5], [x - 7, y + 5], [x - 6, y + 5], [x - 5, y + 5]]

    n2 = [[x - 4, y + 8], [x - 3, y + 8], [x - 2, y + 8], [x - 1, y + 8],
          [x - 4, y + 7], [x - 3, y + 7], [x - 2, y + 7], [x - 1, y + 7],
          [x - 4, y + 6], [x - 3, y + 6], [x - 2, y + 6], [x - 1, y + 6],
          [x - 4, y + 5], [x - 3, y + 5], [x - 2, y + 5], [x - 1, y + 5]]

    n3 = [[x + 1, y + 8], [x + 2, y + 8], [x + 3, y + 8], [x + 4, y + 8],
          [x + 1, y + 7], [x + 2, y + 7], [x + 3, y + 7], [x + 4, y + 7],
          [x + 1, y + 6], [x + 2, y + 6], [x + 3, y + 6], [x + 4, y + 6],
          [x + 1, y + 5], [x + 2, y + 5], [x + 3, y + 5], [x + 4, y + 5]]

    n4 = [[x + 5, y + 8], [x + 6, y + 8], [x + 7, y + 8], [x + 8, y + 8],
          [x + 5, y + 7], [x + 6, y + 7], [x + 7, y + 7], [x + 8, y + 7],
          [x + 5, y + 6], [x + 6, y + 6], [x + 7, y + 6], [x + 8, y + 6],
          [x + 5, y + 5], [x + 6, y + 5], [x + 7, y + 5], [x + 8, y + 5]]

    n5 = [[x - 8, y + 4], [x - 7, y + 4], [x - 6, y + 4], [x - 5, y + 4],
          [x - 8, y + 3], [x - 7, y + 3], [x - 6, y + 3], [x - 5, y + 3],
          [x - 8, y + 2], [x - 7, y + 2], [x - 6, y + 2], [x - 5, y + 2],
          [x - 8, y + 1], [x - 7, y + 1], [x - 6, y + 1], [x - 5, y + 1]]

    n6 = [[x - 4, y + 4], [x - 3, y + 4], [x - 2, y + 4], [x - 1, y + 4],
          [x - 4, y + 3], [x - 3, y + 3], [x - 2, y + 3], [x - 1, y + 3],
          [x - 4, y + 2], [x - 3, y + 2], [x - 2, y + 2], [x - 1, y + 2],
          [x - 4, y + 1], [x - 3, y + 1], [x - 2, y + 1], [x - 1, y + 1]]

    n7 = [[x + 1, y + 4], [x + 2, y + 4], [x + 3, y + 4], [x + 4, y + 4],
          [x + 1, y + 3], [x + 2, y + 3], [x + 3, y + 3], [x + 4, y + 3],
          [x + 1, y + 2], [x + 2, y + 2], [x + 3, y + 2], [x + 4, y + 2],
          [x + 1, y + 1], [x + 2, y + 1], [x + 3, y + 1], [x + 4, y + 1]]

    n8 = [[x + 5, y + 4], [x + 6, y + 4], [x + 7, y + 4], [x + 8, y + 4],
          [x + 5, y + 3], [x + 6, y + 3], [x + 7, y + 3], [x + 8, y + 3],
          [x + 5, y + 2], [x + 6, y + 2], [x + 7, y + 2], [x + 8, y + 2],
          [x + 5, y + 1], [x + 6, y + 1], [x + 7, y + 1], [x + 8, y + 1]]

    n9 = [[x - 8, y - 4], [x - 7, y - 4], [x - 6, y - 4], [x - 5, y - 4],
          [x - 8, y - 3], [x - 7, y - 3], [x - 6, y - 3], [x - 5, y - 3],
          [x - 8, y - 2], [x - 7, y - 2], [x - 6, y - 2], [x - 5, y - 2],
          [x - 8, y - 1], [x - 7, y - 1], [x - 6, y - 1], [x - 5, y - 1]]

    n10 = [[x - 4, y - 4], [x - 3, y - 4], [x - 2, y - 4], [x - 1, y - 4],
           [x - 4, y - 3], [x - 3, y - 3], [x - 2, y - 3], [x - 1, y - 3],
           [x - 4, y - 2], [x - 3, y - 2], [x - 2, y - 2], [x - 1, y - 2],
           [x - 4, y - 1], [x - 3, y - 1], [x - 2, y - 1], [x - 1, y - 1]]

    n11 = [[x + 1, y - 4], [x + 2, y - 4], [x + 3, y - 4], [x + 4, y - 4],
           [x + 1, y - 3], [x + 2, y - 3], [x + 3, y - 3], [x + 4, y - 3],
           [x + 1, y - 2], [x + 2, y - 2], [x + 3, y - 2], [x + 4, y - 2],
           [x + 1, y - 1], [x + 2, y - 1], [x + 3, y - 1], [x + 4, y - 1]]

    n12 = [[x + 5, y - 4], [x + 6, y - 4], [x + 7, y - 4], [x + 8, y - 4],
           [x + 5, y - 3], [x + 6, y - 3], [x + 7, y - 3], [x + 8, y - 3],
           [x + 5, y - 2], [x + 6, y - 2], [x + 7, y - 2], [x + 8, y - 2],
           [x + 5, y - 1], [x + 6, y - 1], [x + 7, y - 1], [x + 8, y - 1]]

    n13 = [[x - 8, y - 8], [x - 7, y - 8], [x - 6, y - 8], [x - 5, y - 8],
           [x - 8, y - 7], [x - 7, y - 7], [x - 6, y - 7], [x - 5, y - 7],
           [x - 8, y - 6], [x - 7, y - 6], [x - 6, y - 6], [x - 5, y - 6],
           [x - 8, y - 5], [x - 7, y - 5], [x - 6, y - 5], [x - 5, y - 5]]

    n14 = [[x - 4, y - 8], [x - 3, y - 8], [x - 2, y - 8], [x - 1, y - 8],
           [x - 4, y - 7], [x - 3, y - 7], [x - 2, y - 7], [x - 1, y - 7],
           [x - 4, y - 6], [x - 3, y - 6], [x - 2, y - 6], [x - 1, y - 6],
           [x - 4, y - 5], [x - 3, y - 5], [x - 2, y - 5], [x - 1, y - 5]]

    n15 = [[x + 1, y - 8], [x + 2, y - 8], [x + 3, y - 8], [x + 4, y - 8],
           [x + 1, y - 7], [x + 2, y - 7], [x + 3, y - 7], [x + 4, y - 7],
           [x + 1, y - 6], [x + 2, y - 6], [x + 3, y - 6], [x + 4, y - 6],
           [x + 1, y - 5], [x + 2, y - 5], [x + 3, y - 5], [x + 4, y - 5]]

    n16 = [[x + 5, y - 8], [x + 6, y - 8], [x + 7, y - 8], [x + 8, y - 8],
           [x + 5, y - 7], [x + 6, y - 7], [x + 7, y - 7], [x + 8, y - 7],
           [x + 5, y - 6], [x + 6, y - 6], [x + 7, y - 6], [x + 8, y - 6],
           [x + 5, y - 5], [x + 6, y - 5], [x + 7, y - 5], [x + 8, y - 5]]

    nf = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9 + n10 + n11 + n12 + n13 + n14 + n15 + n16

    return nf

def concatenate_descriptors(frame_desc,frame_2d_kp, gray_frame, old_gray_frame):
    full_frame_desc = []
    for i in range(frame_desc):
        # Generate optical_desc using optical flow in the 16 neighbors of the keypoint(i)
        optical_desc = gen_optical_desc(frame_2d_kp[i][0], frame_2d_kp[i][1], old_gray_frame, gray_frame)
        optical_desc = np.array(optical_desc)
        frame_desc[i] = np.concatenate((frame_desc[i], optical_desc))
        full_frame_desc.append(frame_desc[i])
        return full_frame_desc


def gen_optical_desc(x, y, old_gray_frame, gray_frame):
    optical_desc = []
    neighbors = gen_neighbors(x, y)
    neighbors = np.array
    neighbors = np.float32(neighbors[:, np.newaxis, :])






