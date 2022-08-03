import pytest

from tempfile import TemporaryDirectory
from pathlib import Path

from privacy_games.utils import AttackResults


class TestAttackResults:
    def test_from_guesses_and_labels_valid_values(self):
        with pytest.raises(ValueError):
            AttackResults.from_guesses_and_labels([1, 0, 2], [1, 1, 0])

    def test_from_guesses_and_labels_computation(self):
        r = AttackResults.from_guesses_and_labels([1, 0, 1, 0], [1, 1, 0, 0])
        assert r.FN == 1
        assert r.FP == 1
        assert r.TN == 1
        assert r.TP == 1 

    def test_serialization(self):
        r = AttackResults(FN=92, FP=3, TN=4, TP=5)
        with TemporaryDirectory() as tmpdir:
            r.to_json(Path(tmpdir) / "results.json")
            r2 = AttackResults.from_json(Path(tmpdir) / "results.json")
            assert r == r2

