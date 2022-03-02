import os
from card_recognizer.infra.algo_ops.pipeline import Op


class TextOp(Op):
    """
    Represents a single text processing operation that can be executed.
    Inputs and outputs can be printed and saved.
    """

    def vis_input(self) -> None:
        """
        Print current input.
        """
        print("Input: " + str(self.input))
        print()

    def vis_output(self) -> None:
        """
        Print current output.
        """
        print(self.name + ": " + str(self.output))
        print()

    def save_input(self, out_path: str = ".") -> None:
        """
        Saves current input text to file.

        param out_path: Path to where input file should be saved.
        """
        if self.input is not None:
            outfile = os.path.join(out_path, self.name + "_input.txt")
            with open(outfile, "w") as out_file:
                out_file.write(self.input)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")

    def save_output(self, out_path: str = ".") -> None:
        """
        Saves current output to file.

        param out_path: Path to where output file should be saved.
        """
        if self.output is not None:
            outfile = os.path.join(out_path, self.name + ".txt")
            with open(outfile, "w") as out_file:
                out_file.write(self.output)
        else:
            raise ValueError("Op " + str(self.name) + " has not executed yet.")
