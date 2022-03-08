import functools
import pickle
from typing import List, Tuple, Callable, Optional, Union

import numpy as np

from card_recognizer.classifier.rules import classify_l1
from card_recognizer.infra.paraloop import paraloop as paraloop


class WordClassifier:
    """
    Classify a card based on detected word frequencies.
    """

    def __init__(self, ref_pkl_file: str, vect_method: str = "basic"):
        """
        Loads reference and vocab from reference pickle file.
        """
        self.pkl_file = ref_pkl_file
        self.ref_mat, self.vocab, self.cards = pickle.load(open(self.pkl_file, "rb"))
        self.vect_method = vect_method

    def _classify_one(
        self,
        classification_func: Callable,
        include_scores: bool,
        ocr_words: List[str],
    ) -> Tuple[int, float, Optional[np.array]]:
        """
        Classify a single OCR result.

        param classification_func: The classification function to identify card number from the word vector
        param ocr_result: The raw OCR result
        param include_scores: Whether to include scores for all cards

        return:
            card_number_prediction: The predicted card number
            score: The score of top match
            scores: The scores for all cards in reference
        """
        v = self.vocab.vect(words=ocr_words, method=self.vect_method)
        card_number_prediction, score, scores = classification_func(
            v=v, ref_mat=self.ref_mat
        )
        if not include_scores:
            scores = None
        return card_number_prediction, score, scores

    def _classify_multiple(
        self,
        ocr_words: List[List[str]],
        classification_func: Callable = classify_l1,
        include_scores: bool = False,
    ) -> Tuple[np.array, np.array, Optional[List[np.array]]]:
        """
        Classify multiple OCR results.

        return:
            preds: The predicted card number for each classifier result
            scores: The scores of top match
        """
        par_classify_func = functools.partial(
            self._classify_one, classification_func, include_scores
        )
        results = paraloop.loop(func=par_classify_func, params=ocr_words)
        preds = np.zeros((len(ocr_words),), dtype=int)
        scores = np.zeros((len(ocr_words),), dtype=float)
        if include_scores:
            all_scores: Optional[List[np.array]] = list()
        else:
            all_scores = None
        for i, result in enumerate(results):
            preds[i] = result[0]
            scores[i] = result[1]
            if include_scores:
                all_scores.append(result[2])
        return preds, scores, all_scores

    def classify(
        self,
        ocr_words: Union[List[str], List[List[str]]],
        classification_func: Callable = classify_l1,
        include_scores: bool = False,
    ) -> Union[Tuple[int, float, np.array], Tuple[np.array, np.array, List[np.array]]]:
        """
        Classify OCR result(s).

        return:
            pred(s): The predicted card number for each classifier result
            score(s): The scores of top match
        """
        if isinstance(ocr_words, list) and not isinstance(ocr_words[0], list):
            return self._classify_one(
                ocr_words=ocr_words,
                classification_func=classification_func,
                include_scores=include_scores,
            )
        else:
            return self._classify_multiple(
                ocr_words=ocr_words,
                classification_func=classification_func,
                include_scores=include_scores,
            )
