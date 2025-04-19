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

def downsample_image(img):
    scale_percent = 0.6
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dims = (width, height)
    return cv.resize(img, dims, interpolation=cv.INTER_AREA)

def find_primary_palette(downsampled_img):
    # use KMeans
    clt = KMeans(n_clusters=NUM_COLORS)
    clt.fit(downsampled_img.reshape(-1, 3))
    ret = clt.cluster_centers_
    # should be of shape (NUM_COLORS, 3)
    return ret 

def add_complements(palette):
    complements = 255 - palette
    palette = np.vstack((palette, complements))
    return palette

def create_blank_canvas(img_x, img_y):
    canvas = np.zeros((img_x, img_y, 3), np.uint8)
    canvas[:, :] = (255, 255, 255)
    return canvas