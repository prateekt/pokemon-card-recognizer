import tempfile
from enum import Enum
from typing import Optional, Dict, Any

import cv2
import numpy as np
import pytesseract
import easyocr

from card_recognizer.infra.cvops.cvops import Pipeline


class OCRPipeline:

    # OCR Method Enum
    class OCRMethod(Enum):
        PYTESSERACT = 1
        EASYOCR = 2

    @staticmethod
    def _run_pytesseract_ocr(img: np.array) -> str:
        """
        Runs pytesseract OCR on an image.

        :param img: The image object
        :return: The text OCR-ed from image
        """
        return pytesseract.image_to_string(img)

    def _run_easy_ocr(self, img: np.array) -> str:
        if self.easy_ocr_reader is None:
            self.easy_ocr_reader = easyocr.Reader(['en'])
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png') as png:
            cv2.imwrite(png.name, img)
            result = self.easy_ocr_reader.readtext(png.name, detail=0)
        return result

    def _run_ocr(self, img: np.array) -> str:
        if self.ocr_method == self.OCRMethod.PYTESSERACT:
            return self._run_pytesseract_ocr(img=img)
        elif self.ocr_method == self.OCRMethod.EASYOCR:
            return self._run_easy_ocr(img=img)

    def __init__(self, img_pipeline: Optional[Pipeline], ocr_method: OCRMethod):
        self.img_pipeline = img_pipeline
        self.ocr_method = ocr_method
        if self.ocr_method == OCRPipeline.OCRMethod.EASYOCR:
            self.easy_ocr_reader = easyocr.Reader(['en'])
        else:
            self.easy_ocr_reader = None

    def run(self, input_img: np.array,
            ocr_method: Optional[OCRMethod] = None) -> str:
        if ocr_method is not None:
            self.ocr_method = ocr_method
        if self.img_pipeline is not None:
            img = self.img_pipeline.run(input=input_img)
        else:
            img = input_img
        return self._run_ocr(img=img)

    def run_on_img_file(self, file: str, ocr_method: Optional[OCRMethod] = None) -> str:
        if ocr_method is not None:
            self.ocr_method = ocr_method
        if self.img_pipeline is not None:
            img = self.img_pipeline.run_on_img_file(file=file)
        else:
            img = cv2.imread(filename=file)
        return self._run_ocr(img=img)

    def set_img_pipeline_params(self, func_name: str, params: Dict[str, Any]) -> None:
        if self.img_pipeline is None:
            raise ValueError('Cannot set parameters when img_pipeline=None.')
        self.img_pipeline.set_params(func_name=func_name, params=params)

    def save(self, outpath: str = '') -> None:
        if self.img_pipeline is None:
            raise ValueError('Cannot save when img_pipeline=None.')
        self.img_pipeline.save(outpath=outpath)


def basic_ocr_pipeline() -> OCRPipeline:
    ocr_pipeline = OCRPipeline(
        img_pipeline=None, ocr_method=OCRPipeline.OCRMethod.PYTESSERACT
    )
    return ocr_pipeline


def black_text_ocr_pipeline() -> OCRPipeline:
    def invert_black_channel(img: np.array) -> np.array:
        # extract black channel in CMYK color space
        # (after this transformation, it appears white)
        img_float = img.astype(np.float) / 255.
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

    img_pipeline = Pipeline(
        funcs=[invert_black_channel, remove_background, invert_back]
    )
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline, ocr_method=OCRPipeline.OCRMethod.PYTESSERACT)
    return ocr_pipeline


def white_text_ocr_pipeline() -> OCRPipeline:
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

    img_pipeline = Pipeline(
        funcs=[gray_scale, remove_background, invert_back]
    )
    ocr_pipeline = OCRPipeline(
        img_pipeline=img_pipeline, ocr_method=OCRPipeline.OCRMethod.PYTESSERACT)
    return ocr_pipeline


if __name__ == "__main__":
    pipeline = basic_ocr_pipeline()
    file = '../reference/data/vivid_voltage/27_hires.png'
#    white_pipeline.set_img_pipeline_params(func_name='remove_background', params={'lower_lim': 90})
    print(pipeline.run_on_img_file(file, ocr_method=OCRPipeline.OCRMethod.EASYOCR))
#    pipeline.save(outpath='/Users/tandonp/Desktop')
