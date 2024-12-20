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
    scores = test_data.map(lambda t: {"log_score": scorer.compute_logscore(t)}, input_columns="text")
    assert scores["log_score"][0] > scores["log_score"][1]