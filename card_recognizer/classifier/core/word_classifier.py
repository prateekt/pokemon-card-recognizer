import functools
from typing import List, Optional, Union

import numpy as np
from algo_ops.ops.text import TextOp
from algo_ops.paraloop import paraloop
from ocr_ops.framework.op.result.ocr_result import OCRPipelineResult, OCRImageResult

from card_recognizer.classifier.core.card_prediction_result import (
    CardPrediction,
    CardPredictionResult,
)
from card_recognizer.classifier.core.rules import (
    classify_l1,
    classify_shared_words,
    classify_shared_words_rarity,
)
from card_recognizer.reference.core.card_reference import CardReference


class WordClassifier(TextOp):
    """
    Classify a card based on detected word frequencies.
    """

    def __init__(
        self,
        ref_pkl_path: str,
        vect_method: str = "basic",
        classification_method: str = "shared_words",
    ):
        """
        param ref_pkl_path: Path to reference pickled model
        param vect_method: Method used to convert words to vector
        param method: The classification method to identify card number from the word vector
        """
        super().__init__(func=self.classify)

        # load reference and vocab
        self.reference = CardReference.load_from_pickle(pkl_path=ref_pkl_path)
        self.vect_method = vect_method

        # structures to help with inference
        self._norm_word_portions = (
            self.reference.ref_mat.T / self.reference.ref_mat.sum(axis=1)
        ).T
        self._norm_word_rarity = self.reference.ref_mat / self.reference.ref_mat.sum(
            axis=0
        )

        # prepare classification method
        self.classification_method = classification_method
        self.classification_func = None
        self.set_classification_method(method=self.classification_method)

        # define input / output
        self.input: Optional[List[List[str]]] = None
        self.output: Optional[CardPredictionResult] = None

    @staticmethod
    def get_supported_classifier_methods() -> List[str]:
        """
        Obtains supported classification methods.
        """
        return ["l1", "shared_words", "shared_words_rarity"]

    def set_classification_method(self, method: str) -> None:
        """
        Sets up classification rule function to method.

        param method: The classification method to identify card number from the word vector
        """
        if method == "l1":
            self.classification_func = functools.partial(
                classify_l1, self._norm_word_portions
            )
        elif method == "shared_words":
            self.classification_func = functools.partial(
                classify_shared_words, self.reference.ref_mat
            )
        elif method == "shared_words_rarity":
            self.classification_func = functools.partial(
                classify_shared_words_rarity, self._norm_word_rarity
            )
        else:
            raise ValueError("Classification method not supported: " + str(method))

    def _classify_one(
        self,
        include_probs: bool,
        ocr_words: List[str],
    ) -> Optional[CardPrediction]:
        """
        Classify a single OCR result.

        param include_probs: Whether to include probabilities for all cards
        ocr_words: List of input words to classify

        return:
            prediction: CardPrediction object if card was detected or None if no vocab words were detected in image.
        """

        # run prediction
        v = self.reference.vocab.vect(words=ocr_words, method=self.vect_method)
        card_number_prediction, probs = self.classification_func(v=v)
        conf = probs[card_number_prediction]

        # return a no-prediction if there are no detected words to predict from
        if np.sum(v) == 0:
            return None
        else:
            # return prediction
            prediction = CardPrediction(
                card_index_in_reference=card_number_prediction, conf=conf
            )
            if include_probs:
                prediction.all_probs = probs
            else:
                prediction.all_probs = None
            return prediction

    def _classify_multiple(
        self,
        ocr_words: List[List[str]],
        include_probs: bool = False,
        mechanism: str = "pool",
    ) -> CardPredictionResult:
        """
        Classify multiple OCR results.

        ocr_words: List of input words to classify
        param include_probs: Whether to include probabilities for all cards
        param mechanism: Paraloop mechanism to use to produce multiple predictions

        return:
            card_prediction_result: CardPredictionResult object
        """
        par_classify_func = functools.partial(self._classify_one, include_probs)
        raw_card_predictions = paraloop.loop(
            func=par_classify_func, params=ocr_words, mechanism=mechanism
        )
        for i, pred in enumerate(raw_card_predictions):
            if pred is not None:
                pred.frame_index = i
        card_predictions = list(filter(None, raw_card_predictions))
        prediction_result = CardPredictionResult(
            predictions=card_predictions, num_frames=len(raw_card_predictions)
        )
        return prediction_result

    def classify(
        self,
        ocr_results: Union[List[List[str]], List[OCRImageResult], OCRPipelineResult],
        include_probs: bool = False,
        mechanism: str = "pool",
    ) -> CardPredictionResult:
        """
        Classify OCR result(s).

        ocr_words: OCRResult, List of OCR Results or Lists of input words to classify
        param include_probs: Whether to include probabilities for all cards
        param mechanism: Paraloop mechanism to use to produce multiple predictions

        return:
            card_prediction_result: CardPredictionResult object for classification task
        """
        if isinstance(ocr_results, OCRPipelineResult):
            # extract OCRPipelineResult into List[OCRResult]
            input_path = ocr_results.input_path
            ocr_results = ocr_results.ocr_image_results
        else:
            input_path = None
        assert isinstance(ocr_results, list)
        if len(ocr_results) == 0:
            return CardPredictionResult(predictions=[])
        if isinstance(ocr_results[0], OCRImageResult):
            # extract List[OCRResult] into List[List[str]]
            ocr_results = [text_box.words for text_box in ocr_results]
        self.input: List[List[str]] = ocr_results
        assert isinstance(self.input, list) and len(self.input) > 0

        # run classifier on List[List[str]]
        if len(self.input) == 1:
            pred = self._classify_one(
                ocr_words=self.input[0], include_probs=include_probs
            )
            if pred is None:
                rtn = CardPredictionResult(predictions=[])
            else:
                rtn = CardPredictionResult(predictions=[pred])
        else:
            rtn = self._classify_multiple(
                ocr_words=self.input, include_probs=include_probs, mechanism=mechanism
            )
        rtn.reference_set = self.reference.name
        rtn.input_path = input_path
        return rtn
