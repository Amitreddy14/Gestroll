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

def add_slight_shifts(w, h, blurry):
    img_coords = []
    for row_val in range(0, h, STRIDE):
        for col_val in range(0, w, STRIDE):
            # experimented with shift values
            x_slight_shift = random.randint(-1, 2)
            y_slight_shift = random.randint(-1, 1)
            col = x_slight_shift + col_val
            row = y_slight_shift + row_val
            if (col < w and row < h):
                img_coords.append((row, col))
            else:
                img_coords.append((row % h, col % w))

    if not blurry:
        random.shuffle(img_coords)
    return img_coords

def get_colors_representing_pixels(img, img_coords):
    colors = []
    for coord in img_coords:
        colors.append(img[coord[0], coord[1]])
    return colors

def compute_color_probabilities(pixels, palette):
    # use distance.cdist
    # reference: open source project from https://www.programcreek.com/python/?CodeExample=compute+color
    distances = distance.cdist(pixels, palette)
    maxima = np.amax(distances, axis=1)

    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]
    
    return distances