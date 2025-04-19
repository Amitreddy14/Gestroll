import cv2 as cv
import numpy as np

def cartoon_effect(frame, color_change):
    # prepare color
    if color_change:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    img_color = cv.pyrDown(cv.pyrDown(frame))
    for _ in range(3):
        img_color = cv.bilateralFilter(img_color, 9, 9, 7)
    img_color = cv.pyrUp(cv.pyrUp(img_color))    