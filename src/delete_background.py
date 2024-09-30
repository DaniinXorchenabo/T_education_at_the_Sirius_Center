from typing import Any, Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image
from cv2 import Mat
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, unsignedinteger
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def image_to_mask(image_filepath, mask_gen: SamAutomaticMaskGenerator, background_level=1) \
        -> tuple[ndarray, list[dict[str, Any]], ndarray | Any]:
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_gen.generate(image)

    masks = sorted(masks, key=lambda x: x['area'], reverse=False)
    background_parts = background_level
    full_mask = masks[0]['segmentation']
    changed_image = np.concatenate((image.copy(), np.ones_like(image)[:, :, 0:1] * 255), axis=2).astype(np.uint8)
    for i in masks[:-background_parts]:
        full_mask = full_mask | i['segmentation']
    full_mask = ~full_mask
    changed_image[full_mask] = [0, 0, 0, 0]
    return changed_image, masks, image


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    return ax, img


def mask_image_generate(image, masks):
    ax, mask_img = show_anns(masks)
    mask_image = Image.fromarray(
        (mask_img * np.concatenate(
            (image.copy(), np.ones_like(image)[:, :, 0:1] * 255), axis=2)
         .astype(np.uint8)
         ).astype('uint8'), 'RGBA', )
    return mask_image
