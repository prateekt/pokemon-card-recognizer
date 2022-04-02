import functools
import os
import traceback
from typing import List

import requests
from algo_ops.paraloop import paraloop
from natsort import natsorted
from pokemontcgsdk import Card, RestClient


def init_api(api_key: str) -> None:
    """
    Init Pokemon TCG SDK client with API Key.
    """
    RestClient.configure(api_key=api_key)


def query_set_cards(set_name: str) -> List[Card]:
    """
    Queries cards from Pokémon card set. Returns sorted by card number.

    param set_name: The name of the card set

    return:
        cards: List of Card objects in the card set
    """
    cards = Card.where(q='set.name:"' + set_name + '"')
    card_numbers = natsorted([card.number for card in cards])
    cards.sort(key=lambda card: card_numbers.index(card.number))
    return cards


def _download_card_image(out_path: str, card: Card) -> None:
    """
    Downloads a card image

    param out_path: The path to where image files are stored
    param card: The card object
    """
    try:
        url = card.images.large
        file_name = os.path.basename(url)
        outfile = os.path.join(out_path, file_name)
        with open(outfile, "wb") as file_out:
            file_out.write(requests.get(url).content)
    except:
        traceback.print_exc()


def download_card_images(cards: List[Card], out_path: str) -> None:
    """
    Downloads card images to an output path for a particular card set.

    param cards: List of cards objects for the card set
    param out_path: Output path where card images should be stored
    """

    # make path
    os.makedirs(out_path, exist_ok=True)

    # download images
    download_func = functools.partial(_download_card_image, out_path)
    paraloop.loop(func=download_func, params=cards)
