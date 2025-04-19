import cv2 as cv
import numpy as np

def cartoon_effect(frame, color_change):
    # prepare color
    if color_change:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)