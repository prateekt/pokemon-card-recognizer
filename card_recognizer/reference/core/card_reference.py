import re
from typing import List, Dict

import numpy as np
from algo_ops.pickleable_object.pickleable_object import PickleableObject
from pokemontcgsdk import Card

from card_recognizer.classifier.core.card_prediction_result import CardPrediction
from card_recognizer.classifier.core.vocab import Vocab


class CardReference(PickleableObject):
    def __init__(self, cards: List[Card], name: str):
        """
        Data structure to represent the "reference" for Pokemon card listing
        (generally a card set like Fusion Strike).

        param cards: List of card objects
        param name: The name of the set
        """

        # create vocabulary
        self.cards = cards
        card_words = CardReference._extract_set_words(cards=self.cards)
        self.vocab = Vocab(card_words=card_words)
        self.name = name

        # build reference matrix
        self.ref_mat = CardReference._create_reference_matrix(
            card_words=card_words, vocab=self.vocab
        )

    @staticmethod
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

    @staticmethod
    def _extract_words_from_card(card: Card) -> List[str]:
        """
        Extracts relevant words from card object (e.g. card name, what Pokémon the card evolves from,
        attacks information, card rules, artist information, etc.).

        param card: The card object

        return:
            List of words
        """
        words: List[str] = list()
        CardReference._add_words(s=card.name, lst=words)
        if card.evolvesFrom is not None:
            CardReference._add_words(s="evolves from", lst=words)
            CardReference._add_words(s=card.evolvesFrom, lst=words)
        if card.abilities is not None:
            for ability in card.abilities:
                CardReference._add_words(s=ability.name, lst=words)
                CardReference._add_words(s=ability.type, lst=words)
                CardReference._add_words(s=ability.text, lst=words)
        if card.attacks is not None:
            for attack in card.attacks:
                CardReference._add_words(s=attack.name, lst=words)
                CardReference._add_words(s=attack.damage, lst=words)
                CardReference._add_words(s=attack.text, lst=words)
        if card.rules is not None:
            for rule in card.rules:
                CardReference._add_words(s=rule, lst=words)
        if card.retreatCost is not None:
            CardReference._add_words(s="retreat", lst=words)
        CardReference._add_words(s=card.flavorText, lst=words)
        CardReference._add_words(s=card.artist, lst=words)
        return words

    @staticmethod
    def _extract_set_words(cards: List[Card]) -> Dict[int, List[str]]:
        """
        Extracts words for all cards in set.

        param cards: Cards in card set

        return:
            Dict that maps card # -> list of card words
        """
        card_words = {
            i: CardReference._extract_words_from_card(card)
            for i, card in enumerate(cards)
        }
        return card_words

    @staticmethod
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
            ref[card_number, :] = vocab.vect(
                words=card_words[card_number], method="basic"
            )
        return ref

    def lookup_by_index(self, card_index: int) -> Card:
        """
        Looks up a card index in the card reference. Returns the card object.

        param card_index: The card index to look up

        Return:
            card: The Card object corresponding to card index.
        """
        return self.cards[card_index]

    def lookup_card_prediction(self, card_prediction: CardPrediction) -> Card:
        """
        Looks up a card prediction in the reference. Returns the card object.

        param card_prediction: Card prediction object to look up

        Return:
            card: The Card object corresponding to card index.
        """
        return self.cards[card_prediction.card_index_in_reference]
