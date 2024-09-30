from typing import Any, Tuple, List, Dict

import cv2
import numpy as np
from cv2 import Mat
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, unsignedinteger
from numpy._typing import _8Bit
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
