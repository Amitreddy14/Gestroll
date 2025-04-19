import cv2
import mediapipe as mp
import numpy as np

def replace_background(fg, bg):
    bg_image = bg
    frame = fg

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()