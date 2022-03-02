import re
import tempfile
from enum import Enum
from typing import Optional, Dict, Any, List, Union

import cv2
import easyocr
import numpy as np
import pytesseract
from spellchecker import SpellChecker

from card_recognizer.infra.algo_ops.cvops import CVPipeline
from card_recognizer.infra.algo_ops.pipeline import Pipeline
from card_recognizer.infra.algo_ops.textops import TextOp


class OCRPipeline:
    """
    OCR Pipeline supports running various OCR methods on an image to generate text. It supports
    using a CVOps image processing pipeline.
    """

    # OCR Method Enum
    class OCRMethod(Enum):
        PYTESSERACT = 1
        EASYOCR = 2

    @staticmethod
    def _run_pytesseract_ocr(img: np.array) -> str:
        """
        Runs pytesseract OCR on an image.

        param img: Input image object
        return:
            output: Text OCR-ed from image
        """
        return pytesseract.image_to_string(img)

    def _run_easy_ocr(self, img: np.array) -> List[str]:
        """
        Runs easyocr method on input image.

        param img: Input image object
        return:
            output: Text OCR-ed from image
        """
        if self.easy_ocr_reader is None:
            self.easy_ocr_reader = easyocr.Reader(["en"])
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png") as png:
            cv2.imwrite(png.name, img)
            result = self.easy_ocr_reader.readtext(png.name, detail=0)
        return result

    def _run_ocr(self, img: np.array) -> Union[str, List[str]]:
        """
        Runs OCR method on image.

        param img: Input image object
        return:
            output: Text OCR-ed from image
        """
        if self.ocr_method == self.OCRMethod.PYTESSERACT:
            return self._run_pytesseract_ocr(img=img)
        elif self.ocr_method == self.OCRMethod.EASYOCR:
            return self._run_easy_ocr(img=img)

    def __init__(
        self,
        img_pipeline: Optional[CVPipeline],
        ocr_method: OCRMethod,
        text_pipeline: Optional[Pipeline],
    ):
        """
        param img_pipeline: An optional CVOps pre-processing pipeline to run on image before OCR
        param ocr_method: The ocr method to use
        param text_pipeline: An optional TextOps pipeline to post-process OCR text
        """
        self.img_pipeline = img_pipeline
        self.ocr_method = ocr_method
        if self.ocr_method == OCRPipeline.OCRMethod.EASYOCR:
            self.easy_ocr_reader = easyocr.Reader(["en"])
        else:
            self.easy_ocr_reader = None
        self.text_pipeline = text_pipeline

    def run(
        self, input_img: np.array, ocr_method: Optional[OCRMethod] = None
    ) -> List[str]:
        """
        Runs OCR pipeline on input image.

        param input_image: Input image
        param ocr_method: OCR method to use (overrides previous setting)
        return:
            output: Text OCR-ed from image
        """
        if ocr_method is not None:
            self.ocr_method = ocr_method
        if self.img_pipeline is not None:
            img = self.img_pipeline.run(inp=input_img)
        else:
            img = input_img
        text = self._run_ocr(img=img)
        if self.text_pipeline is not None:
            text = self.text_pipeline.run(inp=text)
        return text

    def run_on_img_file(
        self, file: str, ocr_method: Optional[OCRMethod] = None
    ) -> List[str]:
        """
        Runs OCR pipeline on input image file.

        param file: Path to input image file
        param ocr_method: OCR method to use (overrides previous setting)
        return:
            output: Text OCR-ed from image
        """
        if ocr_method is not None:
            self.ocr_method = ocr_method
        if self.img_pipeline is not None:
            img = self.img_pipeline.run_on_img_file(file=file)
        else:
            img = cv2.imread(filename=file)
        text = self._run_ocr(img=img)
        if self.text_pipeline is not None:
            text = self.text_pipeline.run(inp=text)
        return text

    def set_img_pipeline_params(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Fixes parameters of CVOPs processing pipeline.

        param func_name: The function name in CVOPs pipeline
        param params: Dict mapping function param -> value
        """
        if self.img_pipeline is None:
            raise ValueError("Cannot set parameters when img_pipeline=None.")
        self.img_pipeline.set_params(func_name=func_name, params=params)

    def set_text_pipeline_params(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Fixes parameters of CVOPs processing pipeline.

        param func_name: The function name in CVOPs pipeline
        param params: Dict mapping function param -> value
        """
        if self.text_pipeline is None:
            raise ValueError("Cannot set parameters when text_pipeline=None.")
        self.text_pipeline.set_params(func_name=func_name, params=params)

    def vis(self) -> None:
        """
        Visualizes image preprocessing pipeline.
        """
        if self.img_pipeline is not None:
            self.img_pipeline.vis()
        if self.text_pipeline is not None:
            self.text_pipeline.vis()

    def save(self, out_path: str = "") -> None:
        """
        Saves image pipeline steps to file.

        param out_path: Where files should go
        """
        if self.img_pipeline is None:
            raise ValueError("Cannot save when img_pipeline=None.")
        self.img_pipeline.save(out_path=out_path)


def basic_ocr_pipeline() -> OCRPipeline:
    """
    Initializes basic pytesseract OCR pipeline.
    """
    ocr_pipeline = OCRPipeline(
        img_pipeline=None,
        ocr_method=OCRPipeline.OCRMethod.PYTESSERACT,
        text_pipeline=None,
    )
    return ocr_pipeline


def text_cleaning_pipeline(ocr_method: OCRPipeline.OCRMethod) -> Pipeline:
    """
    Initializes standard text cleaning pipeline.
    """

    def tokenize_text(text: str) -> List[str]:
        # tokenize text into words
        return [w.strip() for w in text.lower().strip().split(" ")]

    def retokenize_text(text: List[str]) -> List[str]:
        all_words: List[str] = list()
        for phrase in text:
            words = tokenize_text(text=phrase)
            all_words.extend(words)
        return all_words

    def strip(words: List[str]) -> List[str]:
        # remove white space / punctuation
        # reduce to only alphanumeric characters
        stripped = [re.sub("[^a-z0-9]+", "", word) for word in words]
        return stripped

    def correct_spelling(words: List[str]) -> List[str]:
        # attempt to spell check and correct words
        # remove misspellings that cannot be corrected
        new_words = [w for w in words]
        spell = SpellChecker()
        incorrect = spell.unknown(words)
        for i, word in enumerate(new_words):
            if word in incorrect:
                correction = spell.correction(word)
                if correction != word:
                    new_words[i] = correction
                else:
                    new_words[i] = None
        new_words = [word for word in new_words if word is not None]
        return new_words

    def check_vocab(words: List[str], vocab: Dict[str, int]) -> List[str]:
        return [word for word in words if word in vocab.keys()]

    if ocr_method == OCRPipeline.OCRMethod.PYTESSERACT:
        pipeline = Pipeline(
            [tokenize_text, strip, correct_spelling, check_vocab], op_class=TextOp
        )
    else:
        pipeline = Pipeline(
            [retokenize_text, strip, correct_spelling, check_vocab], op_class=TextOp
        )
    return pipeline


def black_text_ocr_pipeline(ocr_method: OCRPipeline.OCRMethod) -> OCRPipeline:
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
        text_pipeline=text_cleaning_pipeline(ocr_method=ocr_method),
    )
    return ocr_pipeline


def white_text_ocr_pipeline(ocr_method) -> OCRPipeline:
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
        text_pipeline=text_cleaning_pipeline(ocr_method=ocr_method),
    )
    return ocr_pipeline
