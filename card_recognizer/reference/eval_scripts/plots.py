from typing import Optional, List, Dict

from ezplotly import EZPlotlyPlot
from scipy.stats import entropy
import ezplotly as ep
import ezplotly_bio as epb

from card_recognizer.reference.core.card_reference import CardReference


def plot_word_entropies(
    reference: CardReference, outfile: Optional[str] = None
) -> None:
    """
    Plot world entropies for a particular card reference.

    param set_name: The name of the set
    outfile: File to output figure to
    """
    word_probs = reference.ref_mat / reference.ref_mat.sum(axis=0)
    entropies = [entropy(word_probs[:, i], base=2) for i in range(word_probs.shape[0])]
    epb.rcdf(
        entropies,
        xlabel="Entropy",
        norm=True,
        y_dtick=0.1,
        outfile=outfile,
        title=reference.set_name + ": Word Entropies",
    )


def plot_word_counts(
    references: Dict[str, CardReference], outfile: Optional[str] = None
) -> None:
    """
    Plot word count distribution for sets.

    param outfile: File output figure to
    """
    h: List[Optional[EZPlotlyPlot]] = [None] * len(references.keys())
    for i, set_name in enumerate(references.keys()):
        words_per_card = references[set_name].ref_mat.sum(axis=1)
        h[i] = ep.hist(
            words_per_card,
            xlabel="# of Words in Card",
            name=set_name,
            title="# of Words Per Card",
            histnorm="probability",
        )
    ep.plot_all(
        h,
        panels=[1] * len(references.keys()),
        showlegend=True,
        outfile=outfile,
        suppress_output=True,
    )
