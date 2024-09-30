import enum
import os

from dotenv import load_dotenv

ROOT_PATH = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
DATA_PATCH = os.path.join(ROOT_PATH, 'data')
sam_checkpoint = '/workspace/NN/weights/sam/sam_vit_h_4b8939.pth'
model_type = "vit_h"
device = "cuda"

load_dotenv(os.path.join(ROOT_PATH, 'env', '.env'))


class DescriptionTypes(enum.Enum):
    simple = 'simple'
