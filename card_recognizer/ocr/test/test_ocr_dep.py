import unittest

import easyocr
import pytesseract


class TestOCRDep(unittest.TestCase):

    def test_easy_ocr(self):
        easy_ocr_reader = easyocr.Reader(["en"])
        easy_ocr_reader.readtext(png.name, detail=0)
