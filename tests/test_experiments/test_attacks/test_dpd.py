import pytest
import torch
import numpy as np

from torch import nn
from opacus.optimizers import DPOptimizer
from opacus import GradSampleModule
from scipy import stats

from privacy_estimates.experiments.attacks.dpd import CanaryGradient, CanaryGradientMethod, CanaryTrackingOptimizer


class TestCanaryGradient:
    @pytest.mark.parametrize("norm", [1.0, 2.0])
    @pytest.mark.parametrize("method", [CanaryGradientMethod.DIRAC, CanaryGradientMethod.RANDOM])
    @pytest.mark.parametrize("is_static", [True, False])
    def test_scale(self, method: CanaryGradientMethod, norm: float, is_static: bool):
        shapes = [[(2, 3), (4, 5)], [(6, 7), (8, 9)]]
        cg = CanaryGradient(shapes=shapes, method=method, norm=norm, is_static=is_static)

        dim_shapes = sum(sum(np.prod(p) for p in group) for group in shapes)
        dim_cg = sum(sum(p.numel() for p in group) for group in cg.gradient) 
        assert dim_shapes == dim_cg

        cg_norm_2 = 0.0
        for group in cg.gradient:
            for p in group:
                cg_norm_2 += p.norm(p=2)**2
        assert cg_norm_2 == pytest.approx(norm**2)


class TestCanaryTrackingOptimizer:
    def test_raise_on_wrapped_dp_optimizer(self):
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        canary = CanaryGradient.from_optimizer(optimizer, "dirac", 1.0)

        # no error
        CanaryTrackingOptimizer(optimizer, canary)

        dp_optimizer = DPOptimizer(optimizer, noise_multiplier=0.1, max_grad_norm=1, expected_batch_size=16)

        with pytest.raises(TypeError):
            CanaryTrackingOptimizer(dp_optimizer, canary)

    @pytest.mark.parametrize("norm", [1.0, 2.0])
    @pytest.mark.parametrize("method", [CanaryGradientMethod.DIRAC, CanaryGradientMethod.RANDOM])
    def test_score(self, norm, method):
        # set up dummy model, optimizer and add some gradients
        model = GradSampleModule(nn.Linear(10, 1, bias=False))

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        optimizer.zero_grad()

        x = torch.randn((8,10))
        y = model(x)
        y.mean().backward()

        canary = CanaryGradient.from_optimizer(optimizer, method=method, norm=norm)

        canary_tracking_optimizer = CanaryTrackingOptimizer(optimizer, canary)

        dp_optimizer = DPOptimizer(canary_tracking_optimizer, noise_multiplier=0.0, max_grad_norm=norm, expected_batch_size=16)
        dp_optimizer.step()

        grad = next(iter(model.parameters())).grad.detach()

        score = grad.flatten().dot(canary.gradient[0][0].flatten())

        assert canary_tracking_optimizer.observations[0] == pytest.approx(score.item())

    def get_observation(self, norm: float, method: str, mean: float, std: float, batch_size: int):
        model = nn.Linear(10, 1, bias=False)
        model = GradSampleModule(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        optimizer.zero_grad()

        x = torch.randn((batch_size, 10)) * std + mean
        y = model(x)
        y.mean().backward()

        canary = CanaryGradient.from_optimizer(optimizer, method=method, norm=norm)

        canary_tracking_optimizer = CanaryTrackingOptimizer(optimizer, canary)

        # Turn off noise and clipping
        dp_optimizer = DPOptimizer(canary_tracking_optimizer, noise_multiplier=0.0, max_grad_norm=1e4,
                                   expected_batch_size=batch_size)
        dp_optimizer.step()

        return canary_tracking_optimizer.observations[0]

    @pytest.mark.parametrize("norm", [1.0, 2.0])
    def test_obeservation_distribution(self, norm):
        mean = 0.0
        std = 1.0
        num_observations = 1_000
        batch_size = 9

        observations = [
            self.get_observation(norm, method="dirac", mean=mean, std=std, batch_size=batch_size)
            for _
            in range(num_observations)
        ]

        assert np.array(observations).mean() == pytest.approx(mean*norm, abs=0.1)
        assert np.array(observations).std() == pytest.approx(std*norm/np.sqrt(batch_size), abs=0.1)

        # test normal
        _, p = stats.shapiro(observations)

        assert p > 0.05




