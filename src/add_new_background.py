import cv2
import numpy as np
from PIL import Image
from numpy import ndarray


def add_new_background(background_filename: str, image_without_background: ndarray):
    im = Image.open(background_filename)
    im = im.resize(image_without_background.shape[:2][::-1])
    new_background = np.array(im)
    image_without_background[image_without_background[:, :, -1] == 0] = new_background[image_without_background[:, :, -1] == 0]
    return image_without_background