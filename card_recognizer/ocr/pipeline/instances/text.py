import re
from typing import List, Dict

from spellchecker import SpellChecker

from card_recognizer.infra.algo_ops.pipeline import Pipeline
from card_recognizer.infra.algo_ops.textops import TextOp


def _tokenize_text(text: str) -> List[str]:
    # tokenize text into words
    return [w.strip() for w in text.lower().strip().split(" ")]


def _retokenize_text(text: List[str]) -> List[str]:
    # retokenizes text into words
    all_words: List[str] = list()
    for phrase in text:
        words = _tokenize_text(text=phrase)
        all_words.extend(words)
    return all_words


def _strip(words: List[str]) -> List[str]:
    # remove white space / punctuation
    # reduce to only alphanumeric characters
    stripped = [re.sub("[^a-z0-9]+", "", word) for word in words]
    return stripped


def _correct_spelling(words: List[str]) -> List[str]:
    # attempt to spell check and correct words
    # remove misspellings that cannot be corrected
    spell = SpellChecker()
    incorrect = spell.unknown(words)
    new_words: List[str] = list()
    for i, word in enumerate(words):
        if word in incorrect:
            correction = spell.correction(word)
            if correction != word:
                new_words.append(correction)
        else:
            new_words.append(word)
    return new_words


def _check_vocab(words: List[str], vocab: Dict[str, int]) -> List[str]:
    return [word for word in words if word in vocab.keys()]


def basic_text_cleaning_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [_tokenize_text, _strip, _correct_spelling, _check_vocab], op_class=TextOp
    )
    return pipeline


def retokenize_text_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [_retokenize_text, _strip, _correct_spelling, _check_vocab], op_class=TextOp
    )
    return pipeline
