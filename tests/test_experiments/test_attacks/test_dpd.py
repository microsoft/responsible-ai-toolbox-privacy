import pytest
import numpy as np

from privacy_estimates.experiments.attacks.dpd import CanaryGradient, CanaryGradientMethod


class TestCanaryGradient:
    @pytest.mark.parametrize("norm", [1.0, 2.0])
    @pytest.mark.parametrize("method", [CanaryGradientMethod.DIRAC])
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
        
        