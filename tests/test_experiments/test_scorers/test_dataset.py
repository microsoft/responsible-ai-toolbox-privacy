from privacy_estimates.experiments.scorers.dataset.scorer import NGramScorer
from datasets import Dataset


def test_ngram_scorer():
    train_data = Dataset.from_dict({
        "text": ["ab cd ef", "12 34 56"]
    })
    scorer = NGramScorer(n=3, train_data=train_data)

    test_data = Dataset.from_dict({
        "text": ["ab cd ef", "uv wx yz"]
    })
    scores = test_data.map(scorer.compute_logscore, input_columns="text")
    assert scores["log_mi_signal"][0] > scores["log_mi_signal"][1]
