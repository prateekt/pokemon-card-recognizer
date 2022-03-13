import functools
import pickle
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Any, Dict

import numpy as np


class Op(ABC):
    """
    Represents a single algorithm operation that can be executed. Inputs and outputs can be visualized.
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
    def vis(self) -> None:
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

    @staticmethod
    def _format_execution_time_stats(
        execution_times: List[float], num_sf: int = 9
    ) -> str:
        """
        Formats execution time stats.

        param execution_times: List of execution times
        num_sf: The number of significant figures to display

        return
            output: (mean) +/- (std) s/calls
        """
        mean_val = np.mean(execution_times)
        std_val = np.std(execution_times)
        return (
            str(np.round(mean_val, num_sf))
            + " +/- "
            + str(np.round(std_val, num_sf))
            + " s/call"
        )

    def vis_profile(self) -> None:
        """
        Prints execution time statistics of Op.
        """
        print(
            self.name
            + ": "
            + self._format_execution_time_stats(execution_times=self.execution_times)
        )

    def to_pickle(self, out_file: str) -> None:
        """
        Pickles current state of Op to file.
        """
        pickle.dump(self, open(out_file, "wb"))

    @staticmethod
    def load_from_pickle(in_file: str) -> "Op":
        """
        Loads op state from a pickle file.
        """
        op = pickle.load(open(in_file, "rb"))
        assert isinstance(op, Op)
        return op
