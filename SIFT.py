import cv2 as cv
import numpy as np


def to_gray(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return gray

def sift_features(gray_image):
    sift_object = cv.xfeatures2d.SIFT_create()
    kp, descritor = sift_object.detectAndCompute(gray_image, None)

    return kp, descritor

def drawkeyPoits(gray,kp,img, img_name):
    #,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite(str(img_name), img)






