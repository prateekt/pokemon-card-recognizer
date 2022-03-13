import os
from typing import Union

import cv2
import numpy as np
from matplotlib import pyplot as plt

from card_recognizer.infra.algo_ops.ops.op import Op

"""
CVOps is infrastructure to build an OpenCV pipeline as a list of
feed-forward ops that support op-level visualization tools and debugging.
"""


class CVOp(Op):
    """
    Represents a single computer vision operation that can be executed.
    Inputs and outputs can be visualized.
    """

    @staticmethod
    def _pyplot_image(img: np.array, title: str) -> None:
        """
        Helper function to plot image using pyplot.

        param img: Image to plot
        param title: Image title
        """
        rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_im)
        plt.title(title)

    def vis_input(self) -> None:
        """
        Plot current input image using pyplot (jupyter compatible)
        """
        self._pyplot_image(img=self.input, title=self.name)

    def vis(self) -> None:
        """
        Plot current output image using pyplot (jupyter compatible)
        """
        self._pyplot_image(img=self.output, title=self.name)

    def save_input(self, out_path: str = ".") -> None:
        """
        Saves current input image to file.

        param out_path: Path to where input image should be saved.
        """
        if self.output is not None:
            outfile = os.path.join(out_path, self.name + "_input.png")
            cv2.imwrite(outfile, self.input)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")

    def save_output(self, out_path: str = ".") -> None:
        """
        Saves current output image to file.

        param out_path: Path to where output image should be saved.
        """
        if self.output is not None:
            outfile = os.path.join(out_path, self.name + ".png")
            cv2.imwrite(outfile, self.output)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")

    def exec(self, inp: Union[np.array, str]) -> np.array:
        """
        A CV op takes in either the file name of an image or an image,
        performs an operation on the image, and returns as new image.

        param inp: The input
        return
            output: The result of the operation
        """
        if isinstance(inp, str):
            inp = cv2.imread(filename=inp)
        if not isinstance(inp, np.ndarray):
            raise ValueError("Unsupported Input: " + str(inp))
        return super().exec(inp=inp)
