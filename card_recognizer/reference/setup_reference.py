import os

from card_recognizer.infra.api.ptcgsdk import query_set_cards, download_card_images
from card_recognizer.reference.card_reference import build_set_reference


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
    download_images = False

    # loop
    for set_name in card_sets:
        # query cards in set
        print(set_name + ": Querying set...")
        cards = query_set_cards(set_name=set_name)
        set_prefix = set_name.lower().replace(" ", "_")

        # download card images
        if download_images:
            out_images_path = os.path.join(out_folder, set_prefix)
            print(set_name + ": Downloading images...")
            download_card_images(cards=cards, out_path=out_images_path)

        # build card text reference pickle file
        print(set_name + ": building reference...")
        out_pkl_path = os.path.join(out_folder, set_prefix + ".pkl")
        build_set_reference(cards=cards, out_pkl_path=out_pkl_path)


if __name__ == "__main__":
    main()
