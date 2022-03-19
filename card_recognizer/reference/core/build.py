import os
from typing import List

from pokemontcgsdk import Card

from card_recognizer.infra.api.ptcgsdk import query_set_cards, download_card_images
from card_recognizer.reference.core.card_reference import CardReference


class ReferenceBuild:
    @staticmethod
    def supported_card_sets() -> List[str]:
        card_sets = [
            "Vivid Voltage",
            "Darkness Ablaze",
            "Chilling Reign",
            "Evolving Skies",
            "Fusion Strike",
            "Brilliant Stars",
        ]
        return card_sets

    @staticmethod
    def get_path_to_build():
        ref_build_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data"
        )
        return ref_build_folder

    @staticmethod
    def get_set_pkl_path(set_name: str) -> str:
        """
        Get path to reference build pickle file for set. Throws a value error if file does not exist.

        param: set_name

        return:
            Path to ref build pickle file for set
        """
        ref_direc = os.path.dirname(os.path.abspath(__file__))
        pkl_file = set_name.lower().replace(" ", "_") + ".pkl"
        ref_build_path = os.path.join(ref_direc, "data", "ref_build", pkl_file)
        if not os.path.exists(ref_build_path):
            raise ValueError(
                "Reference build not found for set: "
                + str(set_name)
                + ". Has reference been setup? If not, run build.py"
            )
        return ref_build_path

    @staticmethod
    def load(set_name: str) -> CardReference:
        """
        Load built reference for set.
        """
        ref_build_path = ReferenceBuild.get_set_pkl_path(set_name=set_name)
        return CardReference.load_from_pickle(pkl_path=ref_build_path)

    @staticmethod
    def build(download_images: bool = True) -> None:
        """
        Downloads images and builds reference for all card sets.

        param: download_images: Whether to download images or skip if already downloaded
        """

        # loop over sets to build set-specific references
        master_set: List[Card] = list()
        for set_name in ReferenceBuild.supported_card_sets():
            # query cards in set
            print(set_name + ": Querying set...")
            cards = query_set_cards(set_name=set_name)
            master_set += cards
            set_prefix = set_name.lower().replace(" ", "_")

            # download card images
            if download_images:
                out_images_path = os.path.join(
                    ReferenceBuild.get_path_to_build(), "card_images", set_prefix
                )
                print(set_name + ": Downloading images...")
                download_card_images(cards=cards, out_path=out_images_path)

            # build card text reference pickle file
            print(set_name + ": building reference...")
            os.makedirs(
                os.path.join(ReferenceBuild.get_path_to_build(), "ref_build"),
                exist_ok=True,
            )
            out_pkl_path = os.path.join(
                ReferenceBuild.get_path_to_build(), "ref_build", set_prefix + ".pkl"
            )
            reference = CardReference(cards=cards)
            reference.to_pickle(out_pkl_path=out_pkl_path)

        # build master reference
        print("Building master reference..")
        out_pkl_path = os.path.join(
            ReferenceBuild.get_path_to_build(), "ref_build", "master.pkl"
        )
        master_reference = CardReference(cards=master_set)
        master_reference.to_pickle(out_pkl_path=out_pkl_path)


if __name__ == "__main__":
    ReferenceBuild.build()
