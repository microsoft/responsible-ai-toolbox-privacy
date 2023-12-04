import pytest
import numpy as np
from typing import List
from privacy_estimates.experiments.attacks.lira.lira import LiRA
from datasets import Dataset, features


class TestLiRA:
    @pytest.mark.parametrize("median_or_mean", ['median', 'mean'])
    def test_lira_tf(self, median_or_mean):
        num_samples = 2
        num_labels = 3
        logits_in = [np.random.rand(4, num_labels) for _ in range(num_samples)]
        logits_out = [np.random.rand(5, num_labels) for _ in range(num_samples)]
        logits_target = [np.random.rand(num_labels) for _ in range(num_samples)]
        labels = [np.random.randint(0, num_labels) for _ in range(num_samples)]

        stats_features = features.Features({
            "sample_index": features.Value("int64"),
            "split": features.Value("int64"),
            "logits_in": features.Sequence(features.Sequence(features.Value(dtype="float64"))),
            "logits_out": features.Sequence(features.Sequence(features.Value(dtype="float64"))),
            "label": features.Value("int64"),
        })

        challenge_points_stats = Dataset.from_dict(
            mapping = {
                "sample_index": range(num_samples),
                "split": [0]*num_samples,
                "logits_in": [[[l for l in m]  for m in s] for s in logits_in],
                "logits_out": [[[l for l in m]  for m in s] for s in logits_out],
                "label": labels
            },
            features = stats_features
        )
        lira = LiRA.from_dataset(challenge_points_stats, mean_estimator=median_or_mean, fix_variance=False, num_proc=None)

        challenge_points = Dataset.from_dict(
            mapping = {
                "sample_index": range(num_samples),
                "split": [0]*num_samples,
                "logits": logits_target,
                "label": labels
            },
            features = features.Features({
                "sample_index": features.Value("int64"),
                "split": features.Value("int64"),
                "logits": features.Sequence(features.Value(dtype="float64")),
                "label": features.Value("int64"),
            })
        )

        challenge_points = lira.compute_lira_score_for_dataset(challenge_points, column_name="score", num_proc=None)
        scores = challenge_points["score"]

        scores_tf = lira_e2e_tf(logits_in, logits_out, logits_target, labels, median_or_mean=median_or_mean, fix_variance=False)

        np.testing.assert_array_almost_equal(scores, scores_tf, decimal=5)


def lira_e2e_tf(logits_in: List[np.ndarray], logits_out: List[np.ndarray], logits_target: List[np.ndarray], labels: List[int],
                median_or_mean: str, fix_variance: bool):
    from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import advanced_mia as amia
    assert len(logits_in) == len(logits_out) == len(logits_target) == len(labels)
    assert all(l_t.ndim == 1 for l_t in logits_target) == True
    logits_target = [l_t[np.newaxis,:] for l_t in logits_target]

    stats_in = [amia.calculate_statistic(lo, np.array([la]*lo.shape[0]), sample_weight=None, is_logits=True, option='logit')[:,np.newaxis] for lo, la in zip(logits_in, labels)]
    stats_out = [amia.calculate_statistic(lo, np.array([la]*lo.shape[0]), sample_weight=None, is_logits=True, option='logit')[:,np.newaxis] for lo, la in zip(logits_out, labels)]
    stats_target = [amia.calculate_statistic(lo, np.array([la]*lo.shape[0]), sample_weight=None, is_logits=True, option='logit') for lo, la in zip(logits_target, labels)]

    scores = amia.compute_score_lira(stats_target, stats_in, stats_out, 'both', fix_variance=fix_variance, median_or_mean=median_or_mean)

    return scores
