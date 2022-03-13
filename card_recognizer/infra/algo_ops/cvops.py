import math
import os
from typing import List, Callable

import cv2
import numpy as np
from matplotlib import pyplot as plt

from card_recognizer.infra.algo_ops.pipeline import Op, Pipeline

"""
CVOps is infrastructure to build an OpenCV pipeline as a list of
feed-forward ops that support op-level visualization tools and debugging.
"""


def _pyplot_image(img: np.array, title: str) -> None:
    """
    Helper function to plot image using pyplot.

    param img: Image to plot
    param title: Image title
    """
    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_im)
    plt.title(title)


class CVOp(Op):
    """
    Represents a single computer vision operation that can be executed.
    Inputs and outputs can be visualized.
    """

    def vis_input(self) -> None:
        """
        Plot current input image using pyplot (jupyter compatible)
        """
        _pyplot_image(img=self.input, title=self.name)

    def vis(self) -> None:
        """
        Plot current output image using pyplot (jupyter compatible)
        """
        _pyplot_image(img=self.output, title=self.name)

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


class CVPipeline(Pipeline):
    """
    Implementation of an OpenCV Image Processing pipeline.
    """

    @classmethod
    def init_from_funcs(cls, funcs: List[Callable], op_class=CVOp) -> "CVPipeline":
        """
        param funcs: List of pipeline functions that execute serially
            as operations in pipeline.
        param op_class: The subclass of Op that the pipeline uses
        """
        assert op_class is CVOp, "Cannot use non-CVOp in CVPipeline."
        op_class = [op_class for _ in range(len(funcs))]
        ops: List[Op] = list()
        for i, func in enumerate(funcs):
            ops.append(op_class[i](func))
        return cls(ops=ops)

    def run_on_img_file(self, file: str) -> np.array:
        """
        Run pipeline on an input image file.

        param file: Path to input image file

        return:
            output: The output image of the pipeline
        """
        img = cv2.imread(filename=file)
        return self.exec(inp=img)

    def vis(
        self, num_cols: int = 4, fig_width: int = 15, fig_height: int = 6, dpi: int = 80
    ) -> None:
        """
        Plot current output images of each Op using pyplot (jupyter compatible).
        Defaults optimize for Jupyter notebook plotting.

        param num_cols: Number of image columns to display
        param fig_width: Total width of figure
        param fig_height: Total height of figure
        param dpi: DPI of figure
        """
        num_rows = math.ceil((len(self.ops.keys()) + 1) / num_cols)
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        plt_num = 1
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            if i == 0:
                plt.subplot(num_rows, num_cols, plt_num)
                op.vis_input()
                plt.title("Input")
                plt_num += 1
            plt.subplot(num_rows, num_cols, plt_num)
            plt_num += 1
            op.vis()
        plt.show()
