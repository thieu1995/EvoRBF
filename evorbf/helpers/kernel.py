#!/usr/bin/env python
# Created by "Thieu" at 11:21, 29/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC):
    """
    Abstract base class for kernel radial basis functions.
    """
    def __init__(self, **params):
        """
        Initialize Kernel with its specific hyperparameters.

        Args:
            params: Dictionary of hyperparameters specific to the RBF kernel.
        """
        self.params = params

    @abstractmethod
    def compute(self, x, c):
        """
        Compute the RBF kernel value between x and c.

        Args:
            x: Input data point(s), shape [n_samples, n_features] or [n_features].
            c: Center, shape [n_features].

        Returns:
            RBF kernel value(s), scalar or array depending on x.
        """
        pass

    def get_params(self):
        """
        Get the current hyperparameters of the RBF kernel.

        Returns:
            Dictionary of hyperparameters.
        """
        return self.params

    def set_params(self, **kwargs):
        """
        Set hyperparameters for the RBF kernel.

        Args:
            kwargs: Hyperparameters to update.
        """
        self.params.update(kwargs)


class GaussianKernel(Kernel):
    """
    Gaussian radial basis function kernel.
    """
    def __init__(self, sigma=1.0):
        super().__init__(sigma=sigma)

    def compute(self, x, c):
        sigma = self.params['sigma']
        return np.exp(-np.linalg.norm(x - c, axis=1)**2 / (2 * sigma**2))


class MultiquadricKernel(Kernel):
    """
    Multiquadric radial basis function kernel.
    """
    def __init__(self, sigma=1.0):
        super().__init__(sigma=sigma)

    def compute(self, x, c):
        sigma = self.params['sigma']
        return np.sqrt(np.linalg.norm(x - c, axis=1)**2 + sigma**2)


class InverseMultiquadricKernel(Kernel):
    """
    Inverse multiquadric radial basis function kernel.
    """
    def __init__(self, sigma=1.0):
        super().__init__(sigma=sigma)

    def compute(self, x, c):
        sigma = self.params['sigma']
        return 1.0 / np.sqrt(np.linalg.norm(x - c, axis=1)**2 + sigma**2)


class LinearKernel(Kernel):
    """
    Linear radial basis function kernel.
    """
    def compute(self, x, c):
        return np.linalg.norm(x - c, axis=1)
