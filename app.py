#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
from collections import deque

from skimage import img_as_float32

import cv2 as cv
import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

from utils.helpers import *
from effects.selfie_segmentation import segment_selfie
from effects.gen_segmentation import segment_image, get_segmented_object
from effects.point_art import *
from effects.collect_effects import *

import tensorflow as tf
import tensorflow_hub as hub

G_seg_image = None
seg_object = None
pickup_point = None
placement_point = None
G_mask = None
seg_mode = False
selfie_seg_mode = True

selection_modes = {
    "select": 0,
    "drawing": 1,
    "effect": 2,
    "segmentation": 3,
    "panoroma": 4,
    "tunnel": 5,
}

def display_selection_mode(selection_mode, display_text):
    for a_key in selection_modes:
        if (selection_mode == selection_modes["select"]):
            display_text += "1. drawing\n2. graphic effects\n3. segmentation\n4. panaroma\n5. light tunnel\n"
            break
        elif (selection_mode == selection_modes["effect"]):
            text = "1. mural\n2. cartoon\n3. point art\n4. avatar\n"
            display_text = text + display_text
            break
        elif selection_mode == selection_modes[a_key]:
            text = a_key + "\n"
            display_text = text + display_text
            break
    return display_text

def add_text(frame, text):
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 3

    y0, dy = 240, 80
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv.putText(frame, line, (50, y), font, fontScale, color, thickness)

    return frame

def stylization_popup(stylization_model, frame, style_image):
    temp_debug_image = frame
    temp_debug_image = tf.expand_dims(temp_debug_image, 0)
    temp_debug_image = img_as_float32(temp_debug_image)
    temp_debug_image = tf.convert_to_tensor(temp_debug_image)

    img = stylization_model(temp_debug_image, style_image)
    img = np.asarray(img[0][0])
    cv.imshow("stylization", img)

def impressionism_popup(frame):
    impressionism = run_impressionistic_filter(frame, False)
    cv.imshow("impressionism", impressionism) 

def place_segmentation(debug_image):
    if seg_object is not None and pickup_point is not None and placement_point is not None:
        difference = np.array(placement_point) - np.array(pickup_point)
        shift_y = int(difference[1])  # col
        shift_x = int(difference[0])  # row   

        if shift_x > 0:
            start_col = 0
            end_col = debug_image.shape[1] - shift_x
            start_col_debug = shift_x
            end_col_debug = debug_image.shape[1]  
        else:
            start_col = abs(shift_x)
            end_col = debug_image.shape[1]
            start_col_debug = 0
            end_col_debug = debug_image.shape[1] - abs(shift_x)  

        if shift_y < 0:
            start_row = abs(shift_y)
            end_row = debug_image.shape[0]
            start_row_debug = 0
            end_row_debug = debug_image.shape[0] - abs(shift_y)

        else:
            start_row = 0
            end_row = debug_image.shape[0] - abs(shift_y)
            start_row_debug = abs(shift_y)
            end_row_debug = debug_image.shape[0]     

        base_seg = np.zeros(
            (debug_image.shape[0], debug_image.shape[1], 3))
        rel_seg_obj = seg_object[start_row:end_row,
                                 start_col:end_col, :]
        base_seg[start_row_debug:end_row_debug,
                 start_col_debug:end_col_debug, :] = rel_seg_obj
        G_mask_temp = G_mask[start_row:end_row, start_col:end_col]  
        
          

       
        


        
          


