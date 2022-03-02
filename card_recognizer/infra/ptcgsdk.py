import os
from typing import List
import requests
from pokemontcgsdk import Card, RestClient
import functools
from card_recognizer.infra.paraloop import paraloop as paraloop

# set up API key for Pokémon TCG API
API_KEY = "1a83fd9c-f080-4ab1-bd68-aea084f2bdc0"
RestClient.configure(API_KEY)


def query_set_cards(set_name: str) -> List[Card]:
    """
    Queries cards from Pokémon card set.

    param set_name: The name of the card set
    return:
        cards: List of Card objects in the card set
    """
    cards = Card.where(q='set.name:"' + set_name + '"')
    return cards


def _download_card_image(out_path: str, card: Card) -> None:
    """
    Downloads a card image

    param out_path: The path to where image files are stored
    param card: The card object
    """
    url = card.images.large
    file_name = os.path.basename(url)
    outfile = os.path.join(out_path, file_name)
    fout = open(outfile, "wb")
    fout.write(requests.get(url).content)
    fout.close()


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
