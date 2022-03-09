import os
import tempfile
from enum import Enum
from typing import Optional, Dict, Any, List, Union

import cv2
import easyocr
import numpy as np
import pytesseract
from natsort import natsorted

from card_recognizer.infra.algo_ops.cvops import CVPipeline
from card_recognizer.infra.algo_ops.pipeline import Pipeline
from card_recognizer.infra.paraloop import paraloop


# OCR Method Enum
class OCRMethod(Enum):
    PYTESSERACT = 1
    EASYOCR = 2


class OCRPipeline:
    """
    OCR Pipeline supports running various OCR methods on an image to generate text. It supports
    using a CVOps image processing pipeline.
    """

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
        if self.ocr_method == OCRMethod.PYTESSERACT:
            return self._run_pytesseract_ocr(img=img)
        elif self.ocr_method == OCRMethod.EASYOCR:
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
        if self.ocr_method == OCRMethod.EASYOCR:
            self.easy_ocr_reader = easyocr.Reader(["en"])
        else:
            self.easy_ocr_reader = None
        self.text_pipeline = text_pipeline

    def run(
        self, input_img: np.array, ocr_method: Optional[OCRMethod] = None
    ) -> Union[str, List[str]]:
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
    ) -> Union[str, List[str]]:
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

    def run_on_images(self, images_dir: str) -> Union[List[str], List[List[str]]]:
        """
        API to run OCR on a directory of images.

        param files_path: Path to directory of card image files

        return:
            output: List of OCR results
        """
        files = natsorted(
            [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
        )
        results = paraloop.loop(func=self.run_on_img_file, params=files)
        return results
