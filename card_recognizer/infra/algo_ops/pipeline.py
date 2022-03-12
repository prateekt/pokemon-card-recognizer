import functools
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, List, Any, Dict

import numpy as np


class Op(ABC):
    """
    Represents a single algorithm operation that can be executed.
    Inputs and outputs can be visualized.
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
        self.execution_times: List[float] = list()

    def exec(self, inp: Any) -> Any:
        """
        Executes operation function on an input. Is also self-time profiling.

        param inp: The input
        return
            output: The result of the operation
        """
        self.input = inp
        t0 = time.time()
        self.output = self.exec_func(inp)
        tf = time.time()
        elapsed_time = tf - t0
        self.execution_times.append(elapsed_time)
        return self.output

    @abstractmethod
    def vis_input(self) -> None:
        """
        Visualize current input.
        """
        pass

    @abstractmethod
    def vis_output(self) -> None:
        """
        Visualize current output.
        """
        pass

    @abstractmethod
    def save_input(self, out_path: str = ".") -> None:
        """
        Saves current input to file.

        param out_path: Path to where input should be saved.
        """
        pass

    @abstractmethod
    def save_output(self, out_path: str = ".") -> None:
        """
        Saves current output to file.

        param out_path: Path to where output should be saved.
        """
        pass

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Sets parameters of operation.

        param params: Dict that maps parameter name -> parameter value
        """
        self.exec_func = functools.partial(self.func, **params)


class Pipeline:
    """
    A pipeline is list of operations that execute serially on an input.
    The output of the previous pipeline step is the input of the next
    pipeline step.
    """

    def __init__(self, funcs: List[Callable], op_class: Any):
        """
        param funcs: List of pipeline functions that execute serially
            as operations in pipeline.
        param op_class: The subclass of Op that the pipeline uses
        """
        self.ops = OrderedDict()
        for func in funcs:
            self.ops[func.__name__] = op_class(func)
        self.execution_times: List[float] = list()

    def set_params(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Fixes parameters of a function in the pipeline.

        param func_name: The name of the function
        param params: The parameters to fix
        """
        op = self.ops[func_name]
        op.set_params(params=params)

    def run(self, inp: Any) -> Any:
        """
        Run entire pipeline on input.

        param inp: The top-level pipeline input
        return:
            output: The output of the pipeline
        """
        t0 = time.time()
        current_input = inp
        for op_name in self.ops.keys():
            op = self.ops[op_name]
            current_input = op.exec(inp=current_input)
        tf = time.time()
        pipeline_execution_time = tf - t0
        self.execution_times.append(pipeline_execution_time)
        return current_input

    def vis(self) -> None:
        """
        Visualize current outputs of each Op.
        """
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            if i == 0:
                op.vis_input()
            op.vis_output()

    def save(self, out_path: str = ".") -> None:
        """
        Saves pipeline Op outputs to file.

        param out_path: Path to where output should go
        """
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            if i == 0:
                op.save_input(out_path=out_path)
            op.save_output(out_path=out_path)

    def vis_profile(self) -> None:
        print("---Profile---")
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            print(
                op_name
                + ": "
                + self._format_execution_time_stats(execution_times=op.execution_times)
            )
        print(
            "Total pipeline: "
            + self._format_execution_time_stats(execution_times=self.execution_times)
        )
        print("-------------")

    @staticmethod
    def _format_execution_time_stats(
        execution_times: List[float], num_sf: int = 9
    ) -> str:
        mean_val = np.mean(execution_times)
        std_val = np.std(execution_times)
        return (
            str(np.round(mean_val, num_sf))
            + " +/- "
            + str(np.round(std_val, num_sf))
            + " s/call"
        )
