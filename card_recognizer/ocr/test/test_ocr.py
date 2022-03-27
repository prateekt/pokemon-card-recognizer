import os
import unittest

import numpy as np

from card_recognizer.ocr.framework.op.ocr_op import OCROp, OCRMethod
from card_recognizer.ocr.framework.pipeline.ocr_pipeline import OCRPipeline
from card_recognizer.ocr.instances.cv import black_text_cv_pipeline, white_text_cv_pipeline
from card_recognizer.ocr.instances.ocr import basic_ocr_pipeline, basic_ocr_with_text_cleaning_pipeline, \
    black_text_ocr_pipeline, white_text_ocr_pipeline
from card_recognizer.ocr.instances.text import basic_text_cleaning_pipeline, retokenize_text_pipeline


class TestOCR(unittest.TestCase):

    def setUp(self) -> None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.test_image = os.path.join(dir_path, "data", "joy_of_data.png")

    def test_ocr_op(self) -> None:
        """
        Test OCR Op.
        """
        ocr_op = OCROp(ocr_method=OCRMethod.EASYOCR)
        output = ocr_op.exec(self.test_image)
        self.assertEqual(output, ["joy", "of", "data"])
        ocr_op = OCROp(ocr_method=OCRMethod.PYTESSERACT)
        output = ocr_op.exec(self.test_image)
        self.assertEqual(output.strip(), "joy of data")

    def test_ocr_pipeline(self) -> None:
        """
        Test OCR in basic pipeline.
        """
        ocr_pipeline = OCRPipeline(img_pipeline=None, ocr_method=OCRMethod.PYTESSERACT, text_pipeline=None)
        output = ocr_pipeline.exec(self.test_image)
        self.assertEqual(output.strip(), "joy of data")

    def test_ocr_pipeline_with_basic_text_cleaning(self) -> None:
        """
        Test OCR pipeline with basic text cleaning.
        """
        ocr_pipeline = OCRPipeline(img_pipeline=None, ocr_method=OCRMethod.PYTESSERACT,
                                   text_pipeline=basic_text_cleaning_pipeline())
        ocr_pipeline.set_text_pipeline_params("_check_vocab", {'vocab_words': {'joy'}})
        output = ocr_pipeline.exec(self.test_image)
        self.assertEqual(output, ['joy'])

    def test_cvpipeline_instances(self) -> None:
        """
        Test CVPipeline instances.
        """

        # black text pipeline test
        cv_pipeline = black_text_cv_pipeline()
        output = cv_pipeline.exec(self.test_image)
        self.assertTrue(isinstance(output, np.ndarray))

        # white text pipeline test
        cv_pipeline = white_text_cv_pipeline()
        output = cv_pipeline.exec(self.test_image)
        self.assertTrue(isinstance(output, np.ndarray))

    def test_textpipeline_instances(self) -> None:
        """
        Test text cleaning pipeline instances.
        """

        # basic cleaning pipeline
        text_pipeline = basic_text_cleaning_pipeline()
        text_pipeline.set_pipeline_params("_check_vocab", {'vocab_words': {'joy', 'of', 'data'}})
        output = text_pipeline.exec("joy of ***%$## data opfu")
        self.assertEqual(output, ['joy', 'of', 'data'])

        # retokenization pipeline
        text_pipeline = retokenize_text_pipeline()
        text_pipeline.set_pipeline_params("_check_vocab", {'vocab_words': {'joy', 'of', 'data'}})
        output = text_pipeline.exec(["joy of   \n", "***%$## \n" "data\n opfu\n\n\n\n"])
        self.assertEqual(output, ['joy', 'of', 'data'])

    def test_ocr_pipeline_instances(self) -> None:
        """
        Test all OCR pipeline instances.
        """

        # basic pipeline
        p1 = basic_ocr_pipeline()
        output = p1.exec(inp=self.test_image)
        self.assertEqual(output.strip(), 'joy of data')

        # basic pipeline with text cleaning
        p2 = basic_ocr_with_text_cleaning_pipeline(vocab_words={'joy', 'of'})
        output = p2.exec(inp=self.test_image)
        self.assertEqual(output, ['joy', 'of'])

        # black text ocr
        p3 = black_text_ocr_pipeline()
        p3.set_text_pipeline_params("_check_vocab", {'vocab_words': {'joy', 'data', 'of'}})
        output = p3.exec(inp=self.test_image)
        self.assertEqual(output, [])

        # white text ocr
        p4 = white_text_ocr_pipeline()
        p4.set_text_pipeline_params("_check_vocab", {'vocab_words': {'data'}})
        output = p4.exec(inp=self.test_image)
        self.assertEqual(output, ['data'])
