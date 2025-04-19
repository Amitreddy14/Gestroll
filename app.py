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
