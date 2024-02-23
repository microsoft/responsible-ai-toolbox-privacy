""" Implements the white-box attack of the paper "Tight Auditing of Differentially Private Machine Learning"

[1] Nasr, Milad, Jamie Hayes, Thomas Steinke, Borja Balle, Florian Tramèr, Matthew Jagielski, Nicholas Carlini, and Andreas Terzis. "Tight Auditing of Differentially Private Machine Learning." arXiv preprint arXiv:2302.07956 (2023).
"""
import torch
import numpy as np
from torch.optim import Optimizer
from typing import Dict, List, Iterable
from enum import Enum
from warnings import warn


class CanaryGradientMethod(Enum):
    DIRAC = "dirac"
    RANDOM = "random"


class CanaryGradient:
    def __init__(self, shapes: List[List[torch.Size]], method: CanaryGradientMethod, norm: float,
                 is_static: bool = True) -> None:
        """Create a canary gradient.

        Args:
            shapes: List of lists of parameter shapes. The outer list corresponds to the parameter groups, and the inner
                    list corresponds to the parameters in each group.
            method: The method used to generate the canary gradient. Currently only "dirac" is supported.
            norm: The norm of the canary gradient.
            is_static: If True, the canary gradient is generated once and then cached. If False, the canary gradient is
                          generated each time it is accessed.
        """
        self.shapes = shapes
        self.method = CanaryGradientMethod(method)
        self.is_static = is_static
        self._cached_static_gradient = None
        self.device = torch.device("cpu")
        self.norm_squared = norm**2

    def from_optimizer(optimizer: Optimizer, method: str, norm: float, is_static: bool = True) -> "CanaryGradient":
        """Create a canary gradient from an optimizer.

        Args:
            optimizer: The optimizer to create the canary gradient from.
            method: The method used to generate the canary gradient. Currently only "dirac" is supported.
            norm: The norm of the canary gradient.
            is_static: If True, the canary gradient is generated once and then cached. If False, the canary gradient is
                          generated each time it is accessed.
        """
        devices = set(p.device for param_group in optimizer.param_groups for p in param_group["params"])
        if len(devices) != 1:
            raise ValueError("All parameters must be on the same device")
        shapes = [[p.shape for p in param_group["params"]] for param_group in optimizer.param_groups]
        return CanaryGradient(shapes=shapes, method=method, norm=norm, is_static=is_static).to(devices.pop())

    def _scale_gradient(self, g: List[List[torch.Tensor]]) -> List[List[torch.Tensor]]:
        """Scale the gradient to have the correct norm"""
        g_norm_squared = 0.0
        for g_p in g:
            for g_i in g_p:
                g_norm_squared += torch.sum(g_i**2)
        g_scale = torch.sqrt(torch.tensor(self.norm_squared, device=self.device) / g_norm_squared)
        for g_p in g:
            for g_i in g_p:
                g_i *= g_scale
        return g

    def _generate_canary_gradient(self) -> List[List[torch.Tensor]]:
        """Generate a new canary gradient"""
        if self.method == CanaryGradientMethod.DIRAC:
            return self._generate_dirac_canary_gradient()
        elif self.method == CanaryGradientMethod.RANDOM:
            return self._generate_random_canary_gradient()
        else:
            raise ValueError(f"Unsupported canary gradient method: {self.method}")

    def _generate_dirac_canary_gradient(self) -> List[List[torch.Tensor]]:
        """Generate a new canary gradient using the Dirac method"""
        num_params = len(self)
        i_rand_element = torch.randint(num_params, size=(1,))

        g = []
        n_elements_seen = 0
        for shape_group in self.shapes:
            g_group = []
            for shape in shape_group:
                g_i = torch.zeros(shape).to(self.device)
                if n_elements_seen <= i_rand_element < n_elements_seen + np.prod(shape):
                    g_i.view(-1)[i_rand_element-n_elements_seen] = torch.sqrt(
                        torch.tensor(self.norm_squared, device=self.device)
                    )
                g_group.append(g_i)
                n_elements_seen += np.prod(shape)
            g.append(g_group)
        return g

    def _generate_random_canary_gradient(self) -> List[List[torch.Tensor]]:
        """Generate a new canary gradient drawing each element from a normal distribution.
        """
        g = []
        for shape_group in self.shapes:
            g_group = []
            for shape in shape_group:
                g_group.append(torch.randn(shape).to(self.device))
            g.append(g_group)
        g = self._scale_gradient(g)
        return g

    @property
    def gradient(self) -> List[List[torch.Tensor]]:
        """Get the canary gradient"""
        if self.is_static:
            if self._cached_static_gradient is None:
                self._cached_static_gradient = self._generate_canary_gradient()
            return self._cached_static_gradient
        else:
            assert self._cached_static_gradient is None
            return self._generate_canary_gradient()

    def __len__(self) -> int:
        """Compute the dimension of the canary gradient"""
        return sum(np.prod(s) for g in self.shapes for s in g)

    def dot_gradient(self, param_groups: List[Dict[str, Iterable[torch.Tensor]]]) -> torch.Tensor:
        """Compute the dot product between the canary gradient and the gradient of parameter groups

        Args:
            param_groups: The parameter groups to compute the dot product with. Must be compatible with the canary
                          gradient.
        """
        o = 0.0
        for canary_param_group, param_group in zip(self.gradient, param_groups):
            for canary_p, p in zip(canary_param_group, param_group["params"]):
                o += torch.dot(canary_p.view(-1), p.grad.view(-1))
        return o

    def shapes_agree(self, param_groups: List[Dict[str, Iterable[torch.Tensor]]]) -> bool:
        """Check if the shapes of the canary gradient and the parameter groups agree"""
        for shapes_group, param_group in zip(self.shapes, param_groups):
            for shape, p in zip(shapes_group, param_group["params"]):
                if shape != p.shape:
                    return False
        return True

    def to(self, device: torch.device) -> "CanaryGradient":
        """Move the canary gradient to a new device"""
        self.device = device
        if self._cached_static_gradient is not None:
            self._cached_static_gradient = [[p.to(device) for p in g] for g in self._cached_static_gradient]
        return self


class CanaryTrackingOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer, canary_gradient: CanaryGradient):
        """
        Args:
            optimizer: The original optimizer to wrap.
            canary_gradient: The canary gradient to use.
        """
        try:
            from opacus.optimizers import DPOptimizer
            if isinstance(optimizer, DPOptimizer):
                raise TypeError("`CanaryTrackingOptimizer` should be wrapped around the original optimizer, "
                                "not the `DPOptimizer`.")
        except ImportError:
            # opacus not installed, no need to worry about DPOptimizer
            warn("opacus not installed, `CanaryTrackingOptimizer` only works with `DPOptimizer`")
            pass

        # Make sure canary_gradient is compatible with optimizer's parameters
        if not canary_gradient.shapes_agree(optimizer.param_groups):
            raise ValueError("Canary gradient and optimizer parameters do not agree in shape")

        self.canary_gradient = canary_gradient
        self.original_optimizer = optimizer
        self.observations = []

    def step(self, closure=None):
        o = self.canary_gradient.dot_gradient(self.original_optimizer.param_groups).item()

        self.observations.append(o)

        self.original_optimizer.step(closure=closure)

    def zero_grad(self, set_to_none: bool = False):
        self.original_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.original_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.original_optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group: dict) -> None:
        self.original_optimizer.add_param_group(param_group)
        if not self.canary_gradient.shapes_agree(self.param_groups):
            raise ValueError("Canary gradient and optimizer parameters do not agree in shape")

    @property
    def param_groups(self):
        return self.original_optimizer.param_groups

    @property
    def defaults(self):
        return self.original_optimizer.defaults

    @property
    def state(self):
        return self.original_optimizer.state


