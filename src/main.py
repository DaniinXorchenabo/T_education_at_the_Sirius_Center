import os
import sys
import os.path

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import argparse

import openai
from openai import OpenAI
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from src.add_new_background import add_new_background
from src.config import model_type, sam_checkpoint, device, ROOT_PATH
from src.delete_background import image_to_mask
from src.description_generator import description_generator


if __name__ == '__main__':

    backgrounds = {
        "blue": "blue.png",
        'grad': "grad.jpg",
        'green': "green.jpg",
        'pink-white-1': "pink_white_1.jpg",
        'pink-white-2': "pink_white_2.png",
        'white': "white.png",
    }

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input-folder", help="Путь до папки, в которой лежат изображения, которые необходимо обработать", required=True)
    parser.add_argument( "--output-folder", help="Путь до папки, в которую нужно положить результаты обработки", required=True)
    parser.add_argument( "--add-background", help=f"Один из {list(backgrounds.keys())} или путь до файла с задним фоном", default="blue")


    args = parser.parse_args()

    if os.path.isdir(args.input_folder) is False:
        raise FileExistsError(f"Указанная папка {args.input_folder} не существует")
    all_files = [os.path.join(args.input_folder, entry.name)
                 for entry in os.scandir(args.input_folder)
                 if entry.is_file()
                 and any(str(entry.name).lower().endswith(i)
                         for i in  ['.png', '.jpg', '.jpeg'] )]

    if os.path.isdir(args.output_folder) is False:
        raise FileExistsError(f"Указанная папка {args.output_folder} не существует")

    background_filename = args.add_background
    if args.add_background in backgrounds:
        background_filename = os.path.join(ROOT_PATH, "backgrounds", backgrounds[args.add_background])
    if os.path.isfile(background_filename) is False or all(str(background_filename.name).lower().endswith(i) is False
                         for i in  ['.png', '.jpg', '.jpeg'] ):
        raise FileExistsError(f"файла заднего фона {background_filename} не существует или имеет недопустимое расширение")




    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    client = openai.Client(
        api_key=os.environ['CHAT_GPT_PERS_SECRET'],
        # base_url="https://api.proxyapi.ru/openai/v1",
    )

    for image_filename in all_files:
        image_without_background, *_ = image_to_mask(image_filename, mask_generator, background_level=1)
        image_with_new_background = add_new_background(background_filename, image_without_background)
        description = description_generator(image_filename, client, )