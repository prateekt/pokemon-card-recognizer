import pickle
import re
from typing import List, Dict, Optional, Tuple

import numpy as np
from pokemontcgsdk import Card

from card_recognizer.reference.vocab import Vocab


def _add_words(s: str, lst: List[str]) -> None:
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
    if card.evolvesFrom is not None:
        _add_words(s="evolves from", lst=words)
        _add_words(s=card.evolvesFrom, lst=words)
    if card.abilities is not None:
        for ability in card.abilities:
            _add_words(s=ability.name, lst=words)
            _add_words(s=ability.type, lst=words)
            _add_words(s=ability.text, lst=words)
    if card.attacks is not None:
        for attack in card.attacks:
            _add_words(s=attack.name, lst=words)
            _add_words(s=attack.damage, lst=words)
            _add_words(s=attack.text, lst=words)
    if card.rules is not None:
        for rule in card.rules:
            _add_words(s=rule, lst=words)
    if card.retreatCost is not None:
        _add_words(s="retreat", lst=words)
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
    card_words = {i: _extract_words_from_card(card) for i, card in enumerate(cards)}
    return card_words


def _create_reference_matrix(
    card_words: Dict[int, List[str]], vocab: Vocab
) -> np.array:
    """
    Creates reference matrix of size (number of cards in set, number of vocab words) where ref[i,j] is the count in
    the ith card of the jth vocab word.

    param card_words: Dictionary of extracted card words mapping card # -> list of card words
    param vocab: Dict that maps vocab word -> word index in vocabulary

    return:
        ref: `np.array[int]` reference cards x words matrix
    """
    ref = np.zeros((len(card_words), vocab.size), dtype=int)
    for card_number in card_words.keys():
        ref[card_number, :] = vocab.vect(words=card_words[card_number], method="basic")
    return ref


def build_set_reference(
    cards: List[Card], out_pkl_path: Optional[str] = None
) -> Tuple[np.array, Vocab]:
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
    vocab = Vocab(card_words=card_words)

    # build reference matrix
    ref = _create_reference_matrix(card_words=card_words, vocab=vocab)

    # pickle output (if needed)
    if out_pkl_path is not None:
        pickle.dump((ref, vocab, cards), open(out_pkl_path, "wb"))

    # return
    return ref, vocab
