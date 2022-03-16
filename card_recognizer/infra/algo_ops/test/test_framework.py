import os
import shutil
import unittest

from card_recognizer.infra.algo_ops.ops.op import Op
from card_recognizer.infra.algo_ops.ops.text import TextOp
from card_recognizer.infra.algo_ops.pipeline.pipeline import Pipeline


class TestFramework(unittest.TestCase):

    # test funcs
    @staticmethod
    def reverse(s: str) -> str:
        return s[::-1]

    @staticmethod
    def append_a(s: str) -> str:
        return s + "a"

    @staticmethod
    def append_b(s: str) -> str:
        return s + "b"

    @staticmethod
    def append_something(s: str, something: str) -> str:
        return s + something

    def test_text_op(self) -> None:
        """
        Test an atomic TextOp.
        """

        # create operation from reverse function
        op = TextOp(func=self.reverse)
        self.assertEqual(op.exec_func, self.reverse)
        self.assertEqual(op.name, "reverse")
        self.assertEqual(op.input, None)
        self.assertEqual(op.output, None)
        self.assertEqual(op.execution_times, [])

        # test op execution
        output = op.exec(inp="ab")
        self.assertEqual(output, "ba")
        self.assertEqual(output, self.reverse(s="ab"))
        self.assertEqual(op.input, "ab")
        self.assertEqual(op.output, "ba")
        self.assertEqual(len(op.execution_times), 1)

        # test op pickle and recover state
        op.to_pickle(out_file="test.pkl")
        reloaded_op = TextOp.load_from_pickle(in_file="test.pkl")
        self.assertEqual(op.input, reloaded_op.input)
        self.assertEqual(op.output, reloaded_op.output)
        self.assertEqual(op.execution_times, reloaded_op.execution_times)
        os.unlink("test.pkl")

        # test op execution again
        output = op.exec(inp="a")
        self.assertEqual(output, "a")
        self.assertEqual(output, self.reverse(s="a"))
        self.assertEqual(op.input, "a")
        self.assertEqual(op.output, "a")
        self.assertEqual(len(op.execution_times), 2)

        # test op pickle and recover state
        op.to_pickle(out_file="test.pkl")
        reloaded_op = TextOp.load_from_pickle(in_file="test.pkl")
        self.assertEqual(op.input, reloaded_op.input)
        self.assertEqual(op.output, reloaded_op.output)
        self.assertEqual(op.execution_times, reloaded_op.execution_times)
        os.unlink("test.pkl")

    def test_pipeline_framework(self) -> None:
        """
        Tests a series of TextOps in the pipeline framework.
        """

        # construct pipeline and check that Ops exist and have empty IO buffers
        pipeline = Pipeline.init_from_funcs(
            funcs=[self.append_a, self.append_b, self.reverse, self.reverse],
            op_class=TextOp,
        )
        self.assertEqual(pipeline.input, None)
        self.assertEqual(pipeline.output, None)
        expected_op_names = ["append_a", "append_b", "reverse", "reverse"]
        expected_funcs = [self.append_a, self.append_b, self.reverse, self.reverse]
        for i, op_name in enumerate(pipeline.ops.keys()):
            op = pipeline.ops[op_name]
            self.assertTrue(isinstance(op, Op))
            self.assertTrue(isinstance(op, TextOp))
            op_hash_name = str(op) + ":" + expected_op_names[i]
            self.assertEqual(op_name, op_hash_name)
            self.assertEqual(op.name, expected_op_names[i])
            self.assertEqual(op.input, None)
            self.assertEqual(op.output, None)
            self.assertEqual(op.exec_func, expected_funcs[i])

        # test running pipeline and check ops IO buffers post execution
        final_output = pipeline.exec(inp="g")
        self.assertEqual(final_output, "gab")
        op_names = list(pipeline.ops.keys())
        for i, op_name in enumerate(op_names):
            op = pipeline.ops[op_name]
            if i == 0:
                self.assertEqual(op.input, "g")
            else:
                self.assertEqual(op.input, pipeline.ops[op_names[i - 1]].output)
            if i == len(op_names) - 1:
                self.assertEqual(op.output, final_output)
            else:
                self.assertEqual(op.output, pipeline.ops[op_names[i + 1]].input)

        # pickle pipeline state
        pipeline.to_pickle(out_file="test.pkl")

        # test running pipeline again and check ops IO buffers post execution
        final_output = pipeline.exec(inp="a")
        self.assertEqual(final_output, "aab")
        op_names = list(pipeline.ops.keys())
        for i, op_name in enumerate(op_names):
            op = pipeline.ops[op_name]
            if i == 0:
                self.assertEqual(op.input, "a")
            else:
                self.assertEqual(op.input, pipeline.ops[op_names[i - 1]].output)
            if i == len(op_names) - 1:
                self.assertEqual(op.output, final_output)
            else:
                self.assertEqual(op.output, pipeline.ops[op_names[i + 1]].input)

        # pipeline vis test
        pipeline.vis()
        pipeline.vis_profile()

        # test reloading pipeline after first and check reloaded pipeline
        reloaded_pipeline = Pipeline.load_from_pickle(in_file="test.pkl")
        assert isinstance(reloaded_pipeline, Pipeline)
        op_names = list(reloaded_pipeline.ops.keys())
        for i, op_name in enumerate(op_names):
            op = reloaded_pipeline.ops[op_name]
            if i == 0:
                self.assertEqual(op.input, "g")
            else:
                self.assertEqual(
                    op.input, reloaded_pipeline.ops[op_names[i - 1]].output
                )
            if i == len(op_names) - 1:
                self.assertEqual(op.output, "gab")
            else:
                self.assertEqual(
                    op.output, reloaded_pipeline.ops[op_names[i + 1]].input
                )

        # clean
        os.unlink("test.pkl")

    def test_parameter_fixing(self) -> None:
        """
        Test fixing a parameter in a function.
        """

        pipeline = Pipeline.init_from_funcs(
            funcs=[self.append_a, self.append_something, self.reverse, self.reverse],
            op_class=TextOp,
        )
        pipeline.set_pipeline_params(
            func_name="append_something", params={"something": "b"}
        )
        self.assertEqual(pipeline.input, None)
        self.assertEqual(pipeline.output, None)
        expected_op_names = ["append_a", "append_something", "reverse", "reverse"]
        for i, op_name in enumerate(pipeline.ops.keys()):
            op = pipeline.ops[op_name]
            self.assertTrue(isinstance(op, Op))
            self.assertTrue(isinstance(op, TextOp))
            op_hash_name = str(op) + ":" + expected_op_names[i]
            self.assertEqual(op_name, op_hash_name)
            self.assertEqual(op.name, expected_op_names[i])
            self.assertEqual(op.input, None)
            self.assertEqual(op.output, None)

    @staticmethod
    def _fake_gen_ans_and_compare(inp: str, pred: str) -> bool:
        correct_ans = inp + "abc"
        return pred == correct_ans

    def test_evaluator(self) -> None:
        """
        Test evaluator capability.
        """

        # init pipeline
        pipeline = Pipeline.init_from_funcs(
            funcs=[self.append_a, self.append_b, self.reverse, self.reverse],
            op_class=TextOp,
        )

        # evaluate on a set of inputs. The pipeline should get it all wrong
        # and produce debug pickle files that can be loaded.
        inputs = ["a", "b", "cc"]
        pipeline.evaluate(
            inputs=inputs,
            eval_func=self._fake_gen_ans_and_compare,
            incorrect_pkl_path="bad_pkl",
        )
        self.assertTrue(os.path.exists("bad_pkl"))
        for inp in inputs:
            self.assertTrue(os.path.exists("bad_pkl/" + inp + ".pkl"))
            reloaded_pipeline = Pipeline.load_from_pickle("bad_pkl/" + inp + ".pkl")
            self.assertTrue(isinstance(reloaded_pipeline, Pipeline))
            self.assertEqual(reloaded_pipeline.input, inp)
        shutil.rmtree("bad_pkl")
