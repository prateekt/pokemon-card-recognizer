import os
import unittest

import easyocr
import pytesseract


class TestOCRDep(unittest.TestCase):

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_image = os.path.join(dir_path, 'data', 'joy_of_data.png')

    def test_easy_ocr(self) -> None:
        """
        Test EasyOCr on sample image.
        """
        easy_ocr_reader = easyocr.Reader(["en"])
        output = easy_ocr_reader.readtext(self.test_image, detail=0)
        self.assertEqual(output, ['joy', 'of', 'data'])

    def test_pytesseract(self) -> None:
        """
        Test PyTesseract on sample image.
        """
        output = pytesseract.image_to_string(self.test_image)
        self.assertEqual(output.strip(), 'joy of data')
