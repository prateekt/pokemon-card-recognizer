import tempfile
from enum import Enum
from typing import List, Union

import cv2
import easyocr
import numpy as np
import pytesseract
from algo_ops.ops.op import Op


# OCR Method Enum
class OCRMethod(Enum):
    PYTESSERACT = 1
    EASYOCR = 2


class OCROp(Op):
    """
    Turns the use of OCR package into an Op. Supports EasyOCR and PyTesseract.
    """

    def save_output(self, out_path: str = ".") -> None:
        pass

    def save_input(self, out_path: str = ".") -> None:
        pass

    def vis(self) -> None:
        pass

    def vis_input(self) -> None:
        pass

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

    def _run_ocr(self, inp: Union[str, np.array]) -> Union[str, List[str]]:
        """
        Runs OCR method on image.

        param img: Either a path to an image or an image object.

        return:
            output: Text OCR-ed from image
        """
        if isinstance(inp, str):
            img = cv2.imread(filename=inp)
        elif isinstance(inp, np.ndarray):
            img = inp
        else:
            raise ValueError("Unsupported input: " + str(inp))
        if self.ocr_method == OCRMethod.PYTESSERACT:
            return self._run_pytesseract_ocr(img=img)
        elif self.ocr_method == OCRMethod.EASYOCR:
            return self._run_easy_ocr(img=img)

    def __init__(self, ocr_method: OCRMethod):
        self.ocr_method = ocr_method
        if self.ocr_method == OCRMethod.EASYOCR:
            self.easy_ocr_reader = easyocr.Reader(["en"])
        else:
            self.easy_ocr_reader = None
        super().__init__(func=self._run_ocr)
