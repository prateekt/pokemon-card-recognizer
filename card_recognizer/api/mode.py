# CardRecognizerPipeline mode enum
from enum import Enum


class Mode(Enum):
    SINGLE_IMAGE = 1
    IMAGE_DIR = 2
    VIDEO = 3
    PULLS_IMAGE_DIR = 4
    PULLS_VIDEO = 5
    BOOSTER_PULLS_IMAGE_DIR = 6
    BOOSTER_PULLS_VIDEO = 7
