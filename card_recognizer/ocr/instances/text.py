import re
from typing import List, Set

from algo_ops.ops.text import TextOp
from algo_ops.pipeline.pipeline import Pipeline
from spellchecker import SpellChecker


def _tokenize_text(text: str) -> List[str]:
    # tokenize text into words
    return [w.strip() for w in text.lower().strip().split(" ") if len(w.strip()) > 0]


def _resplit_new_lines(text: List[str]) -> List[str]:
    new_words: List[str] = list()
    for word in text:
        if "\n" in word:
            new_words.extend(
                [w.strip() for w in word.split("\n") if len(w.strip()) > 0]
            )
        else:
            new_words.append(word)
    return new_words


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


def _check_vocab(words: List[str], vocab_words: Set[str]) -> List[str]:
    return [word for word in words if word in vocab_words]


def basic_text_cleaning_pipeline() -> Pipeline:
    pipeline = Pipeline.init_from_funcs(
        [_tokenize_text, _resplit_new_lines, _strip, _check_vocab], op_class=TextOp
    )
    return pipeline


def retokenize_text_pipeline() -> Pipeline:
    pipeline = Pipeline.init_from_funcs(
        [_retokenize_text, _resplit_new_lines, _strip, _check_vocab],
        op_class=TextOp,
    )
    return pipeline
