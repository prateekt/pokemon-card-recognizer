import cv2
import numpy as np

from card_recognizer.infra.algo_ops.pipeline.cv_pipeline import CVPipeline
from card_recognizer.infra.algo_ops.pipeline.pipeline import Pipeline
from card_recognizer.ocr.pipeline.framework.ocr_pipeline import OCRPipeline, OCRMethod
from card_recognizer.ocr.pipeline.instances.text import (
    basic_text_cleaning_pipeline,
    retokenize_text_pipeline,
)
from card_recognizer.reference.core.vocab import Vocab


def basic_ocr_pipeline() -> OCRPipeline:
    """
    Initializes basic PyTesseract OCR pipeline.
    """
    ocr_pipeline = OCRPipeline(
        img_pipeline=None,
        ocr_method=OCRMethod.PYTESSERACT,
        text_pipeline=None,
    )
    return ocr_pipeline


def basic_ocr_with_text_cleaning_pipeline(
    vocab: Vocab,
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
) -> OCRPipeline:
    """
    Initializes basic PyTesseract pipeline with additional basic text cleaning pipeline.
    """
    img_pipeline = CVPipeline.init_from_funcs(funcs=[_gray_scale])
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline,
        ocr_method=ocr_method,
        text_pipeline=_get_text_cleaning_pipeline(ocr_method=ocr_method),
    )
    ocr_pipeline.set_text_pipeline_params(
        func_name="_check_vocab", params={"vocab": vocab}
    )
    return ocr_pipeline


def _get_text_cleaning_pipeline(ocr_method: OCRMethod) -> Pipeline:
    """
    Choose text cleaning pipeline for OCR method.
    """
    if ocr_method == OCRMethod.PYTESSERACT:
        return basic_text_cleaning_pipeline()
    elif ocr_method == OCRMethod.EASYOCR:
        return retokenize_text_pipeline()


def _invert_black_channel(img: np.array) -> np.array:
    # extract black channel in CMYK color space
    # (after this transformation, it appears white)
    img_float = img.astype(np.float) / 255.0
    k_channel = 1 - np.max(img_float, axis=2)
    k_channel = (255 * k_channel).astype(np.uint8)
    return k_channel


def _gray_scale(img: np.array) -> np.array:
    # convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def _remove_background(img: np.array, lower_lim: int = 190) -> np.array:
    # remove background that is not white
    _, bin_img = cv2.threshold(img, lower_lim, 255, cv2.THRESH_BINARY)
    return bin_img


def _invert_back(img: np.array) -> np.array:
    # Invert back to black text / white background
    inv_img = cv2.bitwise_not(img)
    return inv_img


def black_text_ocr_pipeline(
    ocr_method: OCRMethod = OCRMethod.PYTESSERACT,
) -> OCRPipeline:
    """
    Initializes pipeline to OCR black text.
    """

    img_pipeline = CVPipeline.init_from_funcs(
        funcs=[_invert_black_channel, _remove_background, _invert_back]
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

    img_pipeline = CVPipeline.init_from_funcs(
        funcs=[_gray_scale, _remove_background, _invert_back]
    )
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline,
        ocr_method=ocr_method,
        text_pipeline=_get_text_cleaning_pipeline(ocr_method=ocr_method),
    )
    return ocr_pipeline
