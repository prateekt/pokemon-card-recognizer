import functools
from typing import List, Tuple, Optional, Union

import numpy as np

from card_recognizer.classifier.rules import (
    classify_l1,
    classify_shared_words,
    classify_shared_words_rarity,
)
from card_recognizer.infra.paraloop import paraloop as paraloop
from card_recognizer.reference.card_reference import CardReference


class WordClassifier:
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
        Loads reference and vocab from reference pickle file.
        """
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

    def set_classification_method(self, method: str):
        """
        Prepares classification rule function.
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
    ) -> Tuple[Optional[int], Optional[np.array]]:
        """
        Classify a single OCR result.

        param classification_func: The classification function to identify card number
            from the word vector
        param ocr_result: The raw OCR result
        param include_probs: Whether to include probabilities for all cards

        return:
            card_number_prediction: The predicted card number
            probs: The posterior probabilities for cards
        """

        # run prediction
        v = self.reference.vocab.vect(words=ocr_words, method=self.vect_method)
        card_number_prediction, probs = self.classification_func(v=v)
        if not include_probs:
            probs = None

        # make a no-prediction if there are no detected words to predict from
        if np.sum(v) == 0:
            return None, probs
        else:
            # return prediction
            return card_number_prediction, probs

    def _classify_multiple(
        self,
        ocr_words: List[List[str]],
        include_probs: bool = False,
    ) -> Tuple[List[Optional[int]], Optional[List[np.array]]]:
        """
        Classify multiple OCR results.

        return:
            preds: The predicted card number for each classifier result
            probs: The posterior probabilities for cards
        """
        par_classify_func = functools.partial(self._classify_one, include_probs)
        results = paraloop.loop(func=par_classify_func, params=ocr_words)
        preds: List[Optional[int]] = list()
        if include_probs:
            all_probs: Optional[List[np.array]] = list()
        else:
            all_probs = None
        for i, result in enumerate(results):
            preds.append(result[0])
            if include_probs:
                all_probs.append(result[1])
        return preds, all_probs

    def classify(
        self,
        ocr_words: Union[List[str], List[List[str]]],
        include_probs: bool = False,
    ) -> Union[
        Tuple[Optional[int], np.array],
        Tuple[List[Optional[int]], List[np.array]],
    ]:
        """
        Classify OCR result(s).

        return:
            pred(s): The predicted card number for each classifier result
            score(s): The scores of top match
        """
        if len(ocr_words) == 0:
            return None, None
        elif not isinstance(ocr_words[0], list):
            return self._classify_one(ocr_words=ocr_words, include_probs=include_probs)
        else:
            return self._classify_multiple(
                ocr_words=ocr_words,
                include_probs=include_probs,
            )
