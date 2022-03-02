import functools
from typing import Callable, List, Any, Dict
import numpy as np
import cv2
import os
from collections import OrderedDict

"""
CVOps is infrastructure to build an OpenCV pipeline as a list of feed-forward ops that support op-level 
visualization tools and debugging. 
"""


class Op:
    """
    Represents a single computer vision operation that can be executed. Inputs and outputs can be visualized.
    """

    def __init__(self, func: Callable):
        """
        param func: The operation function
        """
        self.func = func
        self.exec_func = func
        self.name = func.__name__
        self.input = None
        self.output = None

    def exec(self, img: np.array) -> np.array:
        """
        Executes operation function on an input image.

        param img: The input image
        return
            output: The result of the operation
        """
        self.input = img
        self.output = self.exec_func(img=img)
        return self.output

    def save(self, out_path: str = ".") -> None:
        """
        Saves current output image to file.

        param out_path: Path to where output image should be saved.
        """
        if self.output is not None:
            outfile = os.path.join(out_path, self.name + ".png")
            cv2.imwrite(outfile, outfile)
            print(outfile)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Sets parameters of operation.

        param params: Dict that maps parameter name -> parameter value
        """
        self.exec_func = functools.partial(self.func, **params)


class Pipeline:
    """
    A pipeline is list of operations that execute serially on an input image. The output of the previous pipeline
    step is the input of the next pipeline step.
    """

    def __init__(self, funcs: List[Callable]):
        """
        param funcs: List of pipeline functions that execute serially as operations in pipeline.
        """
        self.ops = OrderedDict()
        for func in funcs:
            self.ops[func.__name__] = Op(func=func)

    def set_params(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Fixes parameters of a function in the pipeline.

        param func_name: The name of the function
        param params: The parameters to fix
        """
        op = self.ops[func_name]
        op.set_params(params=params)

    def run_on_img_file(self, file: str) -> np.array:
        """
        Run pipeline on an input image file.

        param file: Path to input image file
        return:
            output: The output image of the pipeline
        """
        img = cv2.imread(filename=file)
        return self.run(img=img)

    def run(self, img: np.array) -> np.array:
        """
        Run pipeline on input image.

        param input: Input image
        return:
            output: The output image of the pipeline
        """
        current_input = img
        for op_name in self.ops.keys():
            op = self.ops[op_name]
            current_input = op.exec(img=current_input)
        return current_input

    def save(self, out_path: str = ".") -> None:
        """
        Saves pipeline Op outputs to file.

        param out_path: Path to where output images should go
        """
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            if i == 0:
                input_file = os.path.join(out_path, "input.png")
                cv2.imwrite(input_file, op.input)
            op.save(out_path=out_path)
