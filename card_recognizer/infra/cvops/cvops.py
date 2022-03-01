import functools
from typing import Callable, List, Any, Dict
import numpy as np
import cv2
import os
from collections import OrderedDict


class Op:

    def __init__(self, func: Callable):
        self.func = func
        self.exec_func = func
        self.name = func.__name__
        self.input = None
        self.output = None

    def exec(self, input: np.array) -> np.array:
        self.input = input
        self.output = self.exec_func(img=input)
        return self.output

    def save(self, outpath: str = '.') -> None:
        if self.output is not None:
            outfile = os.path.join(outpath, self.name + '.png')
            cv2.imwrite(outfile, self.output)
            print(outfile)
        else:
            raise ValueError('Op ' + str(self.name) + ' has not executed yet.')

    def set_params(self, params: Dict[str, Any]) -> None:
        self.exec_func = functools.partial(self.func, **params)


class Pipeline:

    def __init__(self, funcs: List[Callable]):
        self.ops = OrderedDict()
        for func in funcs:
            self.ops[func.__name__] = Op(func=func)

    def set_params(self, func_name: str, params: Dict[str, Any]) -> None:
        op = self.ops[func_name]
        op.set_params(params=params)

    def run_on_img_file(self, file: str) -> np.array:
        img = cv2.imread(filename=file)
        return self.run(input=img)

    def run(self, input: np.array) -> np.array:
        current_input = input
        for op_name in self.ops.keys():
            op = self.ops[op_name]
            current_input = op.exec(input=current_input)
        return current_input

    def save(self, outpath: str = '.') -> None:
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            if i == 0:
                input_file = os.path.join(outpath, 'input.png')
                cv2.imwrite(input_file, op.input)
            op.save(outpath=outpath)
