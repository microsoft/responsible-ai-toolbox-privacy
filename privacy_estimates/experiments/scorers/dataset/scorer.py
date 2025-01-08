import numpy as np
import nltk
from dataclasses import dataclass
from pathlib import Path
from argparse_dataclass import ArgumentParser
from datasets import load_from_disk, Dataset
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends, everygrams
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')


@dataclass
class Arguments:
    synthetic_dataset: Path
    scoring_dataset: Path
    scores: Path
    template: str


class NGramScorer:
    def __init__(self, n: int, train_data: Dataset):
        self.n = n
        self.model = Laplace(n)
        train_texts_tokenized = [word_tokenize(text) for text in train_data["text"]]
        train_texts_padded, padded_vocab = padded_everygram_pipeline(self.n, train_texts_tokenized)
        self.model.fit(train_texts_padded, padded_vocab)

    def compute_logscore(self, text: str):
        """
        Compute the log likelihood of the n-gram model on a given piece of text.
        The log likelihood is the sum of the log likelihood of the n-grams in the text.
        """
        tokens = word_tokenize(text)
        padded = list(pad_both_ends(tokens, n=self.n))
        text_everygrams = list(everygrams(padded, max_len=self.n))
        text_ngrams = [ngram for ngram in text_everygrams if len(ngram) == self.n]
        log_likelihood = np.sum([np.log(self.model.score(ngram[-1], ngram[:-1])) for ngram in text_ngrams])
        return log_likelihood


def main(args: Arguments):
    scoring_ds = load_from_disk(str(args.synthetic_dataset))
    synth_ds = load_from_disk(str(args.scoring_dataset))

    scoring_ds = scoring_ds.map(lambda row: {"text": args.template.format(**row)}, delete_columns=scoring_ds.column_names)
    synth_ds = synth_ds.map(lambda row: {"text": args.template.format(**row)}, delete_columns=synth_ds.column_names)

    scorer = NGramScorer(n=3, train_data=synth_ds)

    scoring_ds = scoring_ds.map(scorer.compute_score)


if __name__ == "__main__":
    parser = ArgumentParser(Arguments)
    args = parser.parse_args()
    main(args=args)