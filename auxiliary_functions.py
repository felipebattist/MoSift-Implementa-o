import cv2 as cv
import numpy as np
import SIFT as sft
import VideoCapture_features as vc
import math
#Lucas Kanade Params
lk_params = dict(winSize = (15,15),
                 maxLevel = 2,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
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

def concatenate_descriptors(count,all_desc,all_2d_points, gray_frame, old_gray_frame):
    i = count
    while i != (len(all_desc)):
        optical_desc = gen_optical_desc(all_2d_points[i][0], all_2d_points[i][0], old_gray_frame, gray_frame)
        optical_desc = np.array(optical_desc)
        all_desc[i] = np.concatenate(([all_desc[i], optical_desc]), axis=None)
        print(all_desc[i])
        i += 1

    return all_desc , i



def gen_optical_desc(x, y, old_gray_frame, gray_frame):
    optical_desc = []
    neighbors = gen_neighbors(x, y)
    neighbors = np.array(neighbors)
    neighbors = np.float32(neighbors[:, np.newaxis, :])

    new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray_frame, gray_frame, neighbors, None, **lk_params)
    histogram = [0,0,0,0,0,0,0,0]
    teste = 0
    for i in range(len(new_points)):
        distance_of_x = new_points[i].T[0]
        distance_of_y = new_points[i].T[1]
        arctg_of_point = np.arctan(distance_of_y/distance_of_x)
        histogram = updates_histogram(arctg_of_point, histogram)
        rest = (i+1)%16
        if (rest) == 0:
            optical_desc += histogram
            teste += 1
            histogram = [0,0,0,0,0,0,0,0]

    return optical_desc




def updates_histogram(arctg, histogram):
    if arctg > (2 * math.pi) / 3 and arctg <= math.pi / 3:
           histogram[0] += 1
    elif arctg > math.pi / 3 and arctg <= math.pi / 6:
           histogram[1] += 1
    elif arctg > math.pi / 6 and arctg <= (11 * math.pi) / 6:
           histogram[2] += 1
    elif arctg > (11 * math.pi) / 6 and arctg <= (5 * math.pi) / 3:
        histogram[3] += 1
    elif arctg > (5 * math.pi) / 3 and arctg <= (4 * math.pi) / 3:
        histogram[4] += 1
    elif arctg > (4 * math.pi) / 3 and arctg <= (5 * math.pi) / 4:
        histogram[5] += 1
    elif arctg > (5 * math.pi) / 4 and arctg <= math.pi:
        histogram[6] += 1
    elif arctg > math.pi and arctg <= (2 * math.pi) / 3:
        histogram[7] += 1
    return histogram








