from collections import OrderedDict
from typing import Callable, List, Any, Dict, Union

from card_recognizer.infra.algo_ops.ops.op import Op


class Pipeline(Op):
    """
    A generic pipeline is list of operations that execute serially on an input.
    The output of the previous pipeline step is the input of the next pipeline step.
    """

    @staticmethod
    def _pipeline_op_name(op: Op) -> str:
        return str(op) + ":" + str(op.name)

    def _run(self, inp: Any) -> Any:
        """
        Run entire pipeline on input, one op at a time.

        param inp: The top-level pipeline input

        return:
            output: The output of the pipeline
        """
        current_input = inp
        for op_name in self.ops.keys():
            op = self.ops[op_name]
            current_input = op.exec(inp=current_input)
        return current_input

    def __init__(self, ops: List[Op]):
        super().__init__(func=self._run)
        self.ops = OrderedDict()
        for i, op in enumerate(ops):
            assert isinstance(op, Op)
            self.ops[self._pipeline_op_name(op=op)] = op

    @classmethod
    def init_from_funcs(
        cls, funcs: List[Callable], op_class: Union[Any, List[Any]]
    ) -> "Pipeline":
        """
        param funcs: List of pipeline functions that execute serially
            as operations in pipeline.
        param op_class: The subclass of Op that the pipeline uses
        """
        if not isinstance(op_class, list):
            op_class = [op_class for _ in range(len(funcs))]
        ops: List[Op] = list()
        for i, func in enumerate(funcs):
            ops.append(op_class[i](func))
        return cls(ops=ops)

    def set_params(self, params: Dict[str, Any]) -> None:
        raise ValueError(
            "Please use set_pipeline_params when setting params of pipeline."
        )

    def _find_op(self, func_name: str) -> Op:
        """
        Helper function to find an Op corresponding to a pipeline function.

        param func_name: Name of function

        return:
            Found Op (or ValueError if no Op found)
        """
        for key in self.ops.keys():
            op = self.ops[key]
            assert isinstance(op, Op)
            if op.name == func_name:
                return op
        raise ValueError("Op not found: " + func_name)

    def set_pipeline_params(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Fixes parameters of a function in the pipeline.

        param func_name: The name of the function
        param params: The parameters to fix
        """
        op = self._find_op(func_name=func_name)
        op.set_params(params=params)

    def save_input(self, out_path: str = ".") -> None:
        raise ValueError("Please use save_output to visualize pipeline data flow.")

    def vis_input(self) -> None:
        raise ValueError("Please use vis to visualize pipeline data flow.")

    def vis(self) -> None:
        """
        Visualize current outputs of each Op.
        """
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            assert isinstance(op, Op)
            if i == 0:
                op.vis_input()
            op.vis()

    def save_output(self, out_path: str = ".") -> None:
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
        """
        Visualizes timing profiling information about pipeline Ops.
        """
        print("---Profile---")
        for i, op_name in enumerate(self.ops.keys()):
            op = self.ops[op_name]
            print(op)
            assert isinstance(op, Op)
            op.vis_profile()
        print(
            "Total: "
            + self._format_execution_time_stats(
                execution_times=list(self.execution_times)
            )
        )
        print("-------------")
