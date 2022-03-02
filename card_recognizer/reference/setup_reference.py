import os

from card_recognizer.reference.card_text_reference import build_set_reference
from card_recognizer.infra.ptcgsdk import download_card_images, query_set_cards


def main():
    """
    Downloads images and builds reference for all card sets.
    """
    out_folder = "data"
    card_sets = [
        "Vivid Voltage",
        "Darkness Ablaze",
        "Chilling Reign",
        "Evolving Skies",
        "Fusion Strike",
        "Brilliant Stars",
    ]
    for set_name in card_sets:
        # query cards in set
        cards = query_set_cards(set_name=set_name)

        # download card images
        set_prefix = set_name.lower().replace(" ", "_")
        out_images_path = os.path.join(out_folder, set_prefix)
        download_card_images(cards=cards, out_path=out_images_path)

        # build card text reference pickle file
        out_pkl_path = os.path.join(out_folder, set_prefix + ".pkl")
        build_set_reference(cards=cards, out_pkl_path=out_pkl_path)


if __name__ == "__main__":
    main()
