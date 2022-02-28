import functools
import os
from typing import Callable, List

import cv2
import numpy as np
import pytesseract
from natsort import natsorted

from card_recognizer.infra.paraloop import paraloop as paraloop


def _run_pytesseract_ocr(img) -> str:
    """
    Runs pytesseract OCR on an image.

    :param img: The image object
    :return: The text OCR-ed from image
    """
    return pytesseract.image_to_string(img)


def _black_text_ocr(img, lower_lim: int = 190) -> str:
    """
    OCR black text on arbitrary non-dark background.

    :param img: The image object
    :return: Extracted black text
    """

    # extract black channel in CMYK color space
    # (after this transformation, it appears white)
    img_float = img.astype(np.float) / 255.
    k_channel = 1 - np.max(img_float, axis=2)
    k_channel = (255 * k_channel).astype(np.uint8)

    # remove background that is not white
    _, bin_img = cv2.threshold(k_channel, lower_lim, 255, cv2.THRESH_BINARY)

    # Invert back to black text / white background
    inv_img = cv2.bitwise_not(bin_img)
    cv2.imwrite('bt_'+str(lower_lim)+'.png', inv_img)

    # algo black text
    return _run_pytesseract_ocr(img=inv_img)


def _white_text_ocr(img, lower_lim: int = 190) -> str:
    """
    OCR white text on arbitrary dark background.

    :param img: The image object
    :return: Extracted white text
    """
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove background that is not white
    _, bin_img = cv2.threshold(gray, lower_lim, 255, cv2.THRESH_BINARY)

    # invert so white text becomes black
    inv_img = cv2.bitwise_not(bin_img)

    # algo black (inverted) text that used to be white
    return _run_pytesseract_ocr(img=inv_img)


def _ocr_bw_text(img) -> str:
    """
    OCR black and white text independently, and then return the concatenated result.

    :return: Extracted black and white text
    """

    # ocr normal image
    final_txt = _run_pytesseract_ocr(img)
    for t in [0, 50, 90, 100, 150, 190, 200, 210, 220, 250]:
        black_text = _black_text_ocr(img=img, lower_lim=t)
        print([str(t), black_text])
        white_text = _white_text_ocr(img=img, lower_lim=t)
        if len(black_text) > len(final_txt):
            final_txt = black_text
        if len(white_text) > len(final_txt):
            final_txt = white_text
    print(final_txt)
    return final_txt


def _ocr_card(ocr_func: Callable, file: str) -> str:
    """
    API to OCR a Pokémon card from file.

    :param ocr_func: The OCR function to use
    :param file: Path to card image file
    :return Extracted text
    """

    # load card image
    img = cv2.imread(filename=file)

    # run algo method
    return ocr_func(img=img)


def ocr_cards(files_path: str, ocr_func: Callable = _ocr_bw_text) -> List[str]:
    """
    API to OCR a list of Pokémon card files.

    :param files_path: Path to directory of card image files
    :param ocr_func: The OCR function to use
    :return: List of OCR results
    """
    files = natsorted([os.path.join(files_path, file) for file in os.listdir(files_path)])
    ocr_func_par = functools.partial(_ocr_card, ocr_func)
    results = paraloop.loop(func=ocr_func_par, params=files)
    return results
