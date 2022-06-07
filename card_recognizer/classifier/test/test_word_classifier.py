import os
import unittest

import ezplotly.settings as plotting_settings
import numpy as np
from algo_ops.dependency.tester_util import clean_paths
from algo_ops.ops.cv import ImageResult
from algo_ops.ops.op import Op
from ocr_ops.framework.struct.ocr_result import OCRResult

from card_recognizer.classifier.core.card_prediction_result import (
    CardPredictionResult,
    CardPrediction,
)
from card_recognizer.classifier.core.word_classifier import WordClassifier
from card_recognizer.reference.core.build import ReferenceBuild
from card_recognizer.run_finding.interval import Interval


class TestWordClassifier(unittest.TestCase):
    @staticmethod
    def _clean_env() -> None:
        clean_paths(
            dirs=("algo_ops_profile",), files=("classify.txt", "classify_input.txt")
        )

    def setUp(self) -> None:

        # suppress plotting
        plotting_settings.SUPPRESS_PLOTS = True

        # check that reference build has been setup
        self.master_model_pkl = ReferenceBuild.get_set_pkl_path(set_name="master")
        self.assertTrue(os.path.exists(self.master_model_pkl))

        # setup input
        self.test_input = ["Charizard", "fire", "burn", "fire", "spin"]
        self._clean_env()

    def tearDown(self) -> None:
        self._clean_env()

    def test_end_to_end(self) -> None:
        """
        Test end to end card prediction capability of WordClassifier.
        """

        # init word classifier
        classifier = WordClassifier(ref_pkl_path=self.master_model_pkl)
        self.assertTrue(isinstance(classifier, Op))
        self.assertEqual(classifier.input, None)
        self.assertEqual(classifier.output, None)
        for method in (
            classifier.vis_profile,
            classifier.save_input,
            classifier.save_output,
        ):
            with self.assertRaises(ValueError):
                method()

        # test with test input List[str]
        output = classifier.exec(inp=[self.test_input])
        self.assertTrue(isinstance(classifier.input, list))
        self.assertTrue(isinstance(classifier.output, CardPredictionResult))
        self.assertEqual(classifier.input, [self.test_input])
        self.assertEqual(output, classifier.output)
        self.assertEqual(output.num_frames, None)
        self.assertEqual(output.reference_set, "master")
        self.assertEqual(output.unique_cards, [755])
        self.assertEqual(len(output), 1)
        self.assertTrue(isinstance(output[0], CardPrediction))
        self.assertEqual(output[0].card_index_in_reference, 755)

        # test vis and save input
        classifier.vis()
        classifier.vis_input()
        classifier.vis_profile()
        classifier.save_input()
        classifier.save_output()
        self.assertTrue(os.path.exists("classify.txt"))
        self.assertTrue(os.path.exists("classify_input.txt"))
        self.assertTrue(
            os.path.exists(os.path.join("algo_ops_profile", "classify.png"))
        )

        # test input wrapped in OCRResult
        input_img = ImageResult(img=np.array([0.0]))
        ocr_result_input = [
            OCRResult.from_text_list(texts=self.test_input, input_img=input_img)
        ]
        output2 = classifier.exec(inp=ocr_result_input)
        self.assertEqual(output2, classifier.output)
        self.assertEqual(output2.num_frames, None)
        self.assertEqual(output2.reference_set, "master")
        self.assertEqual(output2.unique_cards, [755])
        self.assertEqual(len(output2), 1)
        self.assertTrue(isinstance(output2[0], CardPrediction))
        self.assertEqual(output2[0].card_index_in_reference, 755)

    def test_classify_multiple(self) -> None:
        """
        Test classifying a run of length 2 of the same card.
        """
        classifier = WordClassifier(ref_pkl_path=self.master_model_pkl)
        output = classifier.exec(inp=[self.test_input, self.test_input])
        self.assertTrue(isinstance(output, CardPredictionResult))
        self.assertEqual(len(output), 2)
        self.assertEqual(output.unique_cards, [755])
        self.assertEqual(len(output.runs), 1)
        self.assertEqual(output.runs[0].interval, Interval(start=0, end=2))
        self.assertEqual(output.runs[0].card_index, 755)
