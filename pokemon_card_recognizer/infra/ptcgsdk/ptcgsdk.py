import functools
import os
import traceback
from typing import List, Dict

import pokemontcgsdk
import requests
from algo_ops.paraloop import paraloop
from natsort import natsorted
from pokemontcgsdk import Card, RestClient

# sets singleton
available_card_sets = None


def get_available_card_set_names_to_ids() -> Dict[str, str]:
    """
    Returns a dictionary of card set names to card set IDs.
    """
    global available_card_sets
    if available_card_sets is None:
        available_card_sets = {
            card_set.name: card_set.id for card_set in pokemontcgsdk.Set.all()
        }
    return available_card_sets


def init_api(api_key: str) -> None:
    """
    Init Pokémon TCG SDK client with API Key.
    """
    RestClient.configure(api_key=api_key)


def query_set_cards(set_name: str) -> List[Card]:
    """
    Queries cards from Pokémon card set. Returns sorted by card number.

    param set_name: The name of the card set

    return:
        cards: List of Card objects in the card set
    """
    if set_name not in get_available_card_set_names_to_ids():
        raise ValueError("Card set name not found: " + set_name)
    set_id = get_available_card_set_names_to_ids()[set_name]
    cards = Card.where(q='set.id:"' + set_id + '"')
    card_numbers = natsorted([card.number for card in cards])
    cards.sort(key=lambda card: card_numbers.index(card.number))
    return cards


def _download_card_image(out_path: str, num_trials: int, card: Card) -> None:
    """
    Downloads a card image

    param out_path: The path to where image files are stored
    param card: The card object
    """
    trial_num = 0
    url = card.images.large
    while trial_num < num_trials:
        try:
            file_name = os.path.basename(url)
            print("Downloading " + str(file_name))
            outfile = os.path.join(out_path, file_name)
            with open(outfile, "wb") as file_out:
                file_out.write(requests.get(url, timeout=10).content)
            return
        except:
            print("Trial #" + str(trial_num) + " for " + str(url) + " not successful.")
            traceback.print_exc()
            trial_num += 1
    if trial_num == num_trials:
        print("All trials for " + str(url) + " failed.")


def download_card_images(
    cards: List[Card], out_path: str, num_trials: int = 10
) -> None:
    """
    Downloads card images to an output path for a particular card set.

    param cards: List of cards objects for the card set
    param out_path: Output path where card images should be stored
    """

    # make path
    os.makedirs(out_path, exist_ok=True)

    # download images
    download_func = functools.partial(_download_card_image, out_path, num_trials)
    paraloop.loop(func=download_func, params=cards)
