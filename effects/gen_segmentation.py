import numpy as np
import cv2
from keras_segmentation.pretrained import model_from_checkpoint_path

def pspnet_50_ADE_20K(): 
    model_config = {
            "input_height": 473,
            "input_width": 473,
            "n_classes": 150,
            "model_class": "pspnet_50",
            }

    latest_weights  = "model/pspnet50_ade20k.h5"
    
    return model_from_checkpoint_path(model_config, latest_weights)

model = pspnet_50_ADE_20K()  # load the pretrained model trained on ADE20k dataset

def get_segmented_object(seg, img, point):
    color = np.array(seg[point[1], point[0], :])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(seg, color, color)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(img, img, mask=mask)
    return mask, output