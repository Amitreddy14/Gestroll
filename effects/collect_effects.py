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

    # prepare edges
    img_edges = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    img_edges = cv.adaptiveThreshold(
        cv.medianBlur(img_edges, 7), 255,
        cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
        9, 2,)
    img_edges = cv.cvtColor(img_edges, cv.COLOR_GRAY2RGB)