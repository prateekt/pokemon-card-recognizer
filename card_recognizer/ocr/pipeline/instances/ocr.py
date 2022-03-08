import cv2
import numpy as np

from card_recognizer.infra.algo_ops.cvops import CVPipeline
from card_recognizer.ocr.pipeline.instances.text import (
    basic_text_cleaning_pipeline,
    retokenize_text_pipeline,
)
from card_recognizer.ocr.pipeline.ocr_pipeline import OCRPipeline, OCRMethod


def basic_ocr_pipeline() -> OCRPipeline:
    """
    Initializes basic pytesseract OCR pipeline.
    """
    ocr_pipeline = OCRPipeline(
        img_pipeline=None,
        ocr_method=OCRMethod.PYTESSERACT,
        text_pipeline=None,
    )
    return ocr_pipeline


def _get_text_cleaning_pipeline(ocr_method: OCRMethod):
    """
    Choose text cleaning pipeline for OCR method.
    """
    if ocr_method == OCRMethod.PYTESSERACT:
        return basic_text_cleaning_pipeline()
    elif ocr_method == OCRMethod.EASYOCR:
        return retokenize_text_pipeline()


def black_text_ocr_pipeline(
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
) -> OCRPipeline:
    """
    Initializes pipeline to OCR black text.
    """

    def invert_black_channel(img: np.array) -> np.array:
        # extract black channel in CMYK color space
        # (after this transformation, it appears white)
        img_float = img.astype(np.float) / 255.0
        k_channel = 1 - np.max(img_float, axis=2)
        k_channel = (255 * k_channel).astype(np.uint8)
        return k_channel

    def remove_background(img: np.array, lower_lim: int = 190) -> np.array:
        # remove background that is not white
        _, bin_img = cv2.threshold(img, lower_lim, 255, cv2.THRESH_BINARY)
        return bin_img

    def invert_back(img: np.array) -> np.array:
        # Invert back to black text / white background
        inv_img = cv2.bitwise_not(img)
        return inv_img

    img_pipeline = CVPipeline(
        funcs=[invert_black_channel, remove_background, invert_back]
    )
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline,
        ocr_method=ocr_method,
        text_pipeline=_get_text_cleaning_pipeline(ocr_method=ocr_method),
    )
    return ocr_pipeline


def white_text_ocr_pipeline(
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
) -> OCRPipeline:
    """
    Initializes pipeline to OCR white text.
    """

    def gray_scale(img: np.array) -> np.array:
        # convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    def remove_background(img: np.array, lower_lim: int = 190) -> np.array:
        # remove background that is not white
        _, bin_img = cv2.threshold(img, lower_lim, 255, cv2.THRESH_BINARY)
        return bin_img

    def invert_back(img: np.array) -> np.array:
        # invert so white text becomes black
        inv_img = cv2.bitwise_not(img)
        return inv_img

    img_pipeline = CVPipeline(funcs=[gray_scale, remove_background, invert_back])
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline,
        ocr_method=ocr_method,
        text_pipeline=_get_text_cleaning_pipeline(ocr_method=ocr_method),
    )
    return ocr_pipeline
