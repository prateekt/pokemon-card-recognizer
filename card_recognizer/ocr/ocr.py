import functools
import os
from typing import Callable, List

import cv2
import numpy as np
from natsort import natsorted

from card_recognizer.infra.paraloop import paraloop as paraloop
from card_recognizer.ocr import ocr_pipeline

# init pipelines
basic_ocr_pipeline = ocr_pipeline.basic_ocr_pipeline()
black_text_ocr_pipeline = ocr_pipeline.black_text_ocr_pipeline()
white_text_ocr_pipeline = ocr_pipeline.white_text_ocr_pipeline()


def _ocr_bw_text(img: np.array) -> str:
    """
    OCR black and white text independently, and then return the concatenated result.

    param img: Input image
    return:
        output: Extracted text
    """

    # hyper parameters
    lower_lim_settings = [0, 50, 90, 100, 150, 190, 200, 210, 220, 250]

    # run defaults
    basic_text = basic_ocr_pipeline.run(input_img=img)

    # loop over hyper parameters
    final_txt = basic_text
    for lower_lim in lower_lim_settings:
        black_text_ocr_pipeline.set_img_pipeline_params(
            func_name="remove_background", params={"lower_lim": lower_lim}
        )
        black_text = black_text_ocr_pipeline.run(input_img=img)
        white_text_ocr_pipeline.set_img_pipeline_params(
            func_name="remove_background", params={"lower_lim": lower_lim}
        )
        white_text = white_text_ocr_pipeline.run(input_img=img)
        if len(black_text) > len(final_txt):
            final_txt = black_text
        if len(white_text) > len(final_txt):
            final_txt = white_text
    return final_txt


def _ocr_card(ocr_func: Callable, file: str) -> str:
    """
    API to OCR a Pokémon card from file.

    param ocr_func: The OCR function to use
    param file: Path to card image file
    return:
        output: Extracted text
    """

    # load card image
    img = cv2.imread(filename=file)

    # run algo method
    return ocr_func(img=img)


def ocr_cards(files_path: str, ocr_func: Callable = _ocr_bw_text) -> List[str]:
    """
    API to OCR a list of Pokémon card files.

    param files_path: Path to directory of card image files
    param ocr_func: The OCR function to use
    return:
        output: List of OCR results
    """
    files = natsorted(
        [os.path.join(files_path, file) for file in os.listdir(files_path)]
    )
    ocr_func_par = functools.partial(_ocr_card, ocr_func)
    results = paraloop.loop(func=ocr_func_par, params=files)
    return results
