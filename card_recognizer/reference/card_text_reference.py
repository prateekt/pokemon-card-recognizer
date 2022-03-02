import pickle
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
from pokemontcgsdk import Card

from card_recognizer.algo.text_classify import vect_words


def _add_words(s: str, lst: list) -> None:
    """
    Tokenizes and adds words from text string (in-place) to a list of words.

    param s: String containing words
    param lst: Running list of words
    """
    if s is not None:
        # Make string lowercase and remove punctuation
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        if len(s) > 0:
            words = [w.strip() for w in s.split(" ")]
            lst.extend(words)


def _extract_words_from_card(card: Card) -> List[str]:
    """
    Extracts relevant words from card object (e.g. card name, what Pokémon the card evolves from,
    attacks information, card rules, artist information, etc.).

    param card: The card object
    return:
        List of words
    """
    words: List[str] = list()
    _add_words(s=card.name, lst=words)
    _add_words(s=card.evolvesFrom, lst=words)
    if card.attacks is not None:
        for attack in card.attacks:
            _add_words(s=attack.name, lst=words)
            _add_words(s=attack.damage, lst=words)
            _add_words(s=attack.text, lst=words)
    if card.rules is not None:
        for rule in card.rules:
            _add_words(s=rule, lst=words)
    _add_words(s=card.flavorText, lst=words)
    _add_words(s=card.artist, lst=words)
    return words


def _extract_set_words(cards: List[Card]) -> Dict[int, List[str]]:
    """
    Extracts words for all cards in set.

    param cards: Cards in card set
    return:
        Dict that maps card # -> list of card words
    """
    card_words = {card.number: _extract_words_from_card(card) for card in cards}
    return card_words


def _build_vocab(card_words: Dict[int, List[str]]) -> Dict[str, int]:
    """
    Creates vocabulary from card words from all cards in set.

    param card_words: Dict that maps card # in set -> list of card words.
    return:
        vocab: Dict that maps vocab word -> word index in vocabulary
    """
    # create vocab
    vocab: Dict[str, int] = dict()
    i = 0
    for card_name in card_words:
        for word in card_words[card_name]:
            if word not in vocab:
                vocab[word] = i
                i += 1
    return vocab


def _create_reference_matrix(
    card_words: Dict[int, List[str]], vocab: Dict[str, int]
) -> np.array:
    """
    Creates reference matrix of size (number of cards in set, number of vocab words) where ref[i,j] is the count in
    the ith card of the jth vocab word.

    param card_words: Dictionary of extracted card words mapping card # -> list of card words
    param vocab: Dict that maps vocab word -> word index in vocabulary
    return:
        ref: `np.array[int]` reference cards x words matrix
    """
    ref = np.zeros((len(card_words), len(vocab)), dtype=int)
    for i, card_name in enumerate(card_words.keys()):
        ref[i, :] = vect_words(words=card_words[card_name], vocab=vocab)
    return ref


def build_set_reference(
    cards: List[Card], out_pkl_path: Optional[str] = None
) -> Tuple[np.array, Dict[str, int]]:
    """
    API to build reference data structures for Pokémon card set.

    param cards: List of card objects in the set
    param out_pkl_path: The (optional) output pickle file path to store precomputed reference
    return:
        ref: Computed reference cards x words counts matrix
        vocab: Computed vocab dict that maps vocab word -> word index
    """

    # extract card words for each card in set
    card_words = _extract_set_words(cards=cards)

    # create vocabulary
    vocab = _build_vocab(card_words=card_words)

    # build reference matrix
    ref = _create_reference_matrix(card_words=card_words, vocab=vocab)

    # pickle output (if needed)
    if out_pkl_path is not None:
        pickle.dump((ref, vocab, cards), open(out_pkl_path, "wb"))

    # return
    return ref, vocab
