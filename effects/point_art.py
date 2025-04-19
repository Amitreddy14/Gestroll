import numpy as np
import cv2 as cv
import random
from scipy.spatial import distance
from sklearn.cluster import KMeans

# point art constants
RADIUS = 6
NUM_COLORS = 10
THICKNESS = -1
MAX_X = 200
MAX_Y = 200
STRIDE = 4 # better than 2 or 3

def apply_low_pass(img):
    kernel = np.ones((5, 5), np.float32) / 25
    img = cv.filter2D(img, -1, kernel)
    return img