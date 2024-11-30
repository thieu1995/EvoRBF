#!/usr/bin/env python
# Created by "Thieu" at 11:30, 29/11/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from abc import ABC, abstractmethod
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN


class CenterFinder(ABC):
    """
    Abstract base class for center-finding strategies.
    """
    def __init__(self, seed=None, **params):
        """
        Initialize Kernel with its specific hyperparameters.

        Args:
            params: Dictionary of hyperparameters specific to the RBF kernel.
        """
        self.seed = seed
        self.generator = np.random.default_rng(seed)
        self.params = params

    @abstractmethod
    def find_centers(self, X):
        """
        Find centers for the RBF Network.

        Args:
            X: Input data (numpy array of shape [n_samples, n_features]).
            n_centers: Number of centers to find.
            seed: Random seed.

        Returns:
            A numpy array of shape [num_centers, n_features] representing the centers.
        """
        pass


class RandomFinder(CenterFinder):
    """
    Randomly selects centers from the input data.
    """
    def __init__(self, n_centers=10, seed=None, **params):
        super().__init__(seed, **params)
        self.n_centers = n_centers

    def find_centers(self, X):
        indices = self.generator.choice(len(X), self.n_centers, replace=False)
        return X[indices]


class KMeansFinder(CenterFinder):
    """
    Uses k-means clustering to determine the centers.
    """
    def __init__(self, n_centers=10, seed=None, **params):
        super().__init__(seed, **params)
        self.n_centers = n_centers

    def find_centers(self, X):
        kmeans = KMeans(n_clusters=self.n_centers, n_init='auto', random_state=self.seed, **self.params)
        kmeans.fit(X)
        return kmeans.cluster_centers_


class MeanShiftFinder(CenterFinder):
    """
    Uses Mean Shift clustering to determine the centers.
    """
    def __init__(self, bandwidth=2.5, seed=None, **params):
        """
        Args:
            bandwidth: Bandwidth parameter for the mean-shift algorithm.
        """
        super().__init__(seed, **params)
        self.bandwidth = bandwidth

    def find_centers(self, X):
        mean_shift = MeanShift(bandwidth=self.bandwidth, **self.params)
        mean_shift.fit(X)
        return mean_shift.cluster_centers_


class DbscanFinder(CenterFinder):
    """
    Uses DBSCAN clustering to determine the centers.
    """
    def __init__(self, eps=0.75, seed=None, **params):
        """
        Args:
            bandwidth: Bandwidth parameter for the mean-shift algorithm.
        """
        super().__init__(seed, **params)
        self.eps = eps

    def find_centers(self, X):
        mean_shift = DBSCAN(eps=self.eps, min_samples=2, **self.params)
        mean_shift.fit(X)
        return mean_shift.components_
