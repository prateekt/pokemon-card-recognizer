from enum import Enum

"""
Enum representing operating modes for the card recognizer.
"""


class OperatingMode(Enum):
    # process a single image
    SINGLE_IMAGE = 1

    # process a directory of images
    IMAGE_DIR = 2

    # process a video file
    VIDEO = 3

    # process a directory of images representing a video where cards are being shown ("pulled") sequentially
    PULLS_IMAGE_DIR = 4

    # process a video file where cards are being shown ("pulled") sequentially
    PULLS_VIDEO = 5

    # process a directory of images where cards are being shown sequentially, coming from a booster pack.
    BOOSTER_PULLS_IMAGE_DIR = 6

    # process a video file where cards are being shown sequentially, coming from a booster pack.
    BOOSTER_PULLS_VIDEO = 7
