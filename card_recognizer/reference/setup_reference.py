import os
from typing import List

from pokemontcgsdk import Card

from card_recognizer.infra.api.ptcgsdk import query_set_cards, download_card_images
from card_recognizer.reference.card_reference import CardReference


def main():
    """
    Downloads images and builds reference for all card sets.
    """

    # flags
    out_folder = "data"
    card_sets = [
        "Vivid Voltage",
        "Darkness Ablaze",
        "Chilling Reign",
        "Evolving Skies",
        "Fusion Strike",
        "Brilliant Stars",
    ]
    download_images = True

    # loop over sets to build set-specific references
    master_set: List[Card] = list()
    for set_name in card_sets:
        # query cards in set
        print(set_name + ": Querying set...")
        cards = query_set_cards(set_name=set_name)
        master_set += cards
        set_prefix = set_name.lower().replace(" ", "_")

        # download card images
        if download_images:
            out_images_path = os.path.join(out_folder, "card_images", set_prefix)
            print(set_name + ": Downloading images...")
            download_card_images(cards=cards, out_path=out_images_path)

        # build card text reference pickle file
        print(set_name + ": building reference...")
        os.makedirs(os.path.join(out_folder, "ref_build"), exist_ok=True)
        out_pkl_path = os.path.join(out_folder, "ref_build", set_prefix + ".pkl")
        reference = CardReference(cards=cards)
        reference.to_pickle(out_pkl_path=out_pkl_path)

    # build master reference
    print("Building master reference..")
    out_pkl_path = os.path.join(out_folder, "ref_build", "master.pkl")
    master_reference = CardReference(cards=master_set)
    master_reference.to_pickle(out_pkl_path=out_pkl_path)


if __name__ == "__main__":
    main()
