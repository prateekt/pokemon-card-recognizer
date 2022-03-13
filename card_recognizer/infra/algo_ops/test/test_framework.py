import unittest

from card_recognizer.infra.algo_ops.pipeline import Pipeline
from card_recognizer.infra.algo_ops.textops import TextOp


class TestFramework(unittest.TestCase):
    def test_pipeline_framework(self):
        def append_a(s: str) -> str:
            return s + "a"

        def append_b(s: str) -> str:
            return s + "b"

        def reverse(s: str) -> str:
            return s[::-1]

        pipeline = Pipeline.init_from_funcs(
            funcs=[append_a, append_b, reverse, reverse], op_class=TextOp
        )
        final_output = pipeline.exec(inp="g")
        pipeline.vis()
        pipeline.vis_profile()
        print(final_output)
