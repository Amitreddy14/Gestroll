import cv2
import mediapipe as mp
import numpy as np

def replace_background(fg, bg):
    bg_image = bg
    frame = fg