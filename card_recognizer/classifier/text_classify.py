import functools
from typing import List, Dict, Tuple, Callable, Optional, Union
from card_recognizer.infra.paraloop import paraloop as paraloop
import numpy as np


def vect_words(words: List[str], vocab: Dict[str, int]) -> np.array:
    """
    Converts a list of words into a feature vector of word counts.

    param words: List of words
    param vocab: Dict mapping word -> index of word in vector
    return:
        v: `np.array[int]` of counts of ith word in vocab in the input list of words
    """
    v = np.zeros((len(vocab),), dtype=int)
    for word in words:
        if word in vocab:
            v[vocab[word]] += 1
    return v


def vect_words_encapsulation_match(words: List[str], vocab: Dict[str, int]) -> np.array:
    """
    Converts a list of words into a feature vector of word counts. Allows for encapsulation matches of word in noisy
    extracted word.

    param words: List of words
    param vocab: Dict mapping word -> index of word
    return:
        v: `np.array[int]` of counts of ith word in vocab in the input list of words
    """
    v = np.zeros((len(vocab),), dtype=int)
    for word in words:
        if word in vocab:
            v[vocab[word]] += 1
        else:
            for i, vocab_word in enumerate(vocab):
                if vocab_word in word:
                    v[i] += 1
    return v


def classify_l1(v: np.array, ref_mat: np.array) -> Tuple[int, float]:
    """

    Nearest match of word vector (v) to reference matrix based on L1 norm.

    param v: `np.array[int]` Word vector
    param ref_mat: `np.array[int]` Set reference matrix
    return:
        card_number_prediction: The predicted card number
        score: The score of top match
    """
    d = np.sum(np.abs(ref_mat - v), axis=1)
    card_number_prediction = d.argmin()
    score = d[card_number_prediction]
    return card_number_prediction, score


def classify_common_words(v: np.array, ref_mat: np.array) -> Tuple[int, float]:
    """
    Classifies the nearest match of word vector v based on commonly shared words with reference card text.

    param v: `np.array[int]` Word vector
    param ref_mat: `np.array[int]` Set reference matrix
    return:
        card_number_prediction: The predicted card number
        score: The score of top match
    """
    scores = ((ref_mat > 0) & (v > 0)).sum(axis=1)
    card_number_prediction = scores.argmax()
    score = scores[card_number_prediction]
    return card_number_prediction, score


def _classify_one(
    ref_mat: np.array,
    vocab: Dict[str, int],
    vect_func: Callable,
    classification_func: Callable,
    ocr_result: Optional[str],
) -> Tuple[int, float]:
    """
    Classify a single OCR result.

    param ref_mat: The reference matrix for the set
    param vocab: The set vocab
    param vect_func: The vectorization function to convert words to vector
    param classification_func: The classification function to identify card number from the word vector
    param ocr_result: The raw OCR result

    return:
        card_number_prediction: The predicted card number
        score: The score of top match
    """
    ocr_words = [w.strip() for w in ocr_result.lower().split(" ")]
    v = vect_func(ocr_words, vocab)
    card_number_prediction, score = classification_func(v=v, ref_mat=ref_mat)
    return card_number_prediction, score


def _classify_multiple(
    ocr_results: List[str],
    ref_mat: np.array,
    vocab: Dict[str, int],
    vect_func: Callable,
    classification_func: Callable = classify_l1,
) -> Tuple[np.array, np.array]:
    """
    Classify multiple OCR results.

    :return:
        preds: The predicted card number for each classifier result
        scores: The scores of top match
    """
    par_classify_func = functools.partial(
        _classify_one, ref_mat, vocab, vect_func, classification_func
    )
    results = paraloop.loop(func=par_classify_func, params=ocr_results)
    preds = np.zeros((len(ocr_results),), dtype=int)
    scores = np.zeros((len(ocr_results),), dtype=float)
    for i, result in enumerate(results):
        preds[i] = result[0]
        scores[i] = result[1]
    return preds, scores


def classify(
    ocr: Union[str, List[str]],
    ref_mat: np.array,
    vocab: Dict[str, int],
    vect_func: Callable = vect_words_encapsulation_match,
    classification_func: Callable = classify_l1,
) -> Union[Tuple[int, float], Tuple[np.array, np.array]]:
    """
    Classify OCR result(s).

    return:
        pred(s): The predicted card number for each classifier result
        score(s): The scores of top match
    """

    if isinstance(ocr, str):
        return _classify_one(
            ocr_result=ocr,
            ref_mat=ref_mat,
            vocab=vocab,
            vect_func=vect_func,
            classification_func=classification_func,
        )
    else:
        return _classify_multiple(
            ocr_results=ocr,
            ref_mat=ref_mat,
            vocab=vocab,
            vect_func=vect_func,
            classification_func=classification_func,
        )
