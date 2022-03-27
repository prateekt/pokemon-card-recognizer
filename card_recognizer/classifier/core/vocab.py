from typing import Dict, List

import numpy as np
from bidict import bidict


class Vocab:
    """
    Vocab is a bidirectional dict that maps vocab word <-> word index in vector space. It supports the capability to
    take in a list of words and generate a word count vector from the bag of words. Different vectorization rules are
    supported to create a word vector from a listing of words. Different rules help with different noise cases when with
    OCR.
    """

    def __init__(self, card_words: Dict[int, List[str]]):
        self._words: bidict[str, int] = self._build_vocab(card_words=card_words)
        self.size = len(self._words)
        self.word_transforms: bidict[str, str] = self.compute_word_transforms()

    def __setitem__(self, key, val):
        self._words[key] = val

    def __getitem__(self, key):
        return self._words[key]

    def __contains__(self, key):
        return key in self._words

    def __call__(self, *args, **kwargs):
        return set(self._words)

    # noinspection PyUnresolvedReferences
    def inv(self, key):
        return self._words.inv[key]

    @staticmethod
    def _build_vocab(
        card_words: Dict[int, List[str]], min_word_length: int = 2
    ) -> bidict[str, int]:
        """
        Creates vocabulary from card words from all cards in set.

        param card_words: Dict that maps card # in set -> list of card words.
        param min_word_length: The minimum character length for a word

        return:
            vocab: Bidirectional dict that maps vocab word <-> word index in vocabulary
        """
        # create vocab words
        words: bidict[str, int] = bidict()
        i = 0
        for card_name in card_words:
            for word in card_words[card_name]:
                if len(word) > min_word_length and word not in words:
                    words[word] = i
                    i += 1
        return words

    def compute_word_transforms(self):
        """
        Identify related words in vocab such that one word can be transformed into the other according to some
        linguistic rule. For example, if one word is the plural of the other, then the singular -> plural are related
        words.
        """
        word_transforms: bidict[str, str] = bidict()
        for word in self._words.keys():
            if word + "s" in self._words:
                word_transforms[word] = word + "s"
            if word + "ed" in self._words:
                word_transforms[word] = word + "ed"
        return word_transforms

    def _vect_basic(self, words: List[str]) -> np.array:
        """
        Converts a list of words into a feature vector of word counts according to the vocab.

        param words: List of words
        param vocab: Vocab to construct vector from

        return:
            v: `np.array[int]` of counts of ith word in vocab in the input list of words
        """
        v = np.zeros((self.size,), dtype=int)
        for word in words:
            if word in self._words:
                v[self._words[word]] += 1
        return v

    @staticmethod
    def _check_word_boundaries(word: str, vocab_word: str) -> bool:
        """
        Helper function to check word boundaries of a word.
        """
        inx = word.find(vocab_word)
        left_inx = 0
        right_inx = len(word) - 1
        if inx > 0 and inx + 1 < len(word):
            left_inx = inx - 1
            right_inx = inx + 1
        own_word = not word[left_inx].isalpha() and not word[right_inx].isalpha()
        return own_word

    def _vect_words_encapsulation_match(self, words: List[str]) -> np.array:
        """
        Converts a list of words into a feature vector of word counts. Allows for encapsulation matches of word in noisy
        extracted word.

        param words: List of words
        param vocab: Vocab to construct vector from

        return:
            v: `np.array[int]` of counts of ith word in vocab in the input list of words
        """
        v = np.zeros((self.size,), dtype=int)
        for word in words:
            if word in self._words:
                v[self._words[word]] += 1
            else:
                for i, vocab_word in enumerate(self._words.keys()):
                    if vocab_word in word:
                        encompass_word = vocab_word in word
                        not_plural_err = not (
                            vocab_word in self.word_transforms.keys()
                            and self.word_transforms[vocab_word] in word
                        )
                        own_word = self._check_word_boundaries(
                            word=word, vocab_word=vocab_word
                        )
                        if encompass_word and not_plural_err and own_word:
                            v[i] += 1
        return v

    def vect(self, words: List[str], method: str = "basic") -> np.array:
        """
        API to convert a list of words into a vector.

        param words: The list of words
        param method: The vectorization method

        return:
            Vector representation of word list in vocab-space
        """
        if method == "basic":
            return self._vect_basic(words=words)
        elif method == "encapsulation_match":
            return self._vect_words_encapsulation_match(words=words)
        else:
            raise ValueError("Unsupported vectorization method: " + str(method))
