#!/usr/bin/env python
# Created by "Thieu" at 12:36, 17/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from scipy.stats import boxcox, yeojohnson
from scipy.special import inv_boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


class OneHotEncoder:
    """
    A simple implementation of one-hot encoding for 1D categorical data.

    Attributes:
        categories_ (np.ndarray): Sorted array of unique categories fitted from the input data.
    """
    def __init__(self):
        """Initialize the encoder with no categories."""
        self.categories_ = None

    def fit(self, X):
        """
        Fit the encoder to the unique categories in X.

        Args:
            X (array-like): 1D array of categorical values.

        Returns:
            self: Fitted OneHotEncoder instance.
        """
        X = np.asarray(X).ravel()
        self.categories_ = np.unique(X)
        return self

    def transform(self, X):
        """
        Transform input data into one-hot encoded format.

        Args:
            X (array-like): 1D array of categorical values.

        Returns:
            np.ndarray: One-hot encoded array of shape (n_samples, n_categories).

        Raises:
            ValueError: If the encoder has not been fitted or unknown category is found.
        """
        if self.categories_ is None:
            raise ValueError("The encoder has not been fitted yet.")

        X = np.asarray(X).ravel()
        one_hot = np.zeros((X.shape[0], len(self.categories_)), dtype=int)

        for i, val in enumerate(X):
            indices = np.where(self.categories_ == val)[0]
            if len(indices) == 0:
                raise ValueError(f"Unknown category encountered during transform: {val}")
            one_hot[i, indices[0]] = 1
        return one_hot

    def fit_transform(self, X):
        """
        Fit the encoder to X and transform X.

        Args:
            X (array-like): 1D array of categorical values.

        Returns:
            np.ndarray: One-hot encoded array of shape (n_samples, n_categories).
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, one_hot):
        """
        Convert one-hot encoded data back to original categories.

        Args:
            one_hot (np.ndarray): 2D array of one-hot encoded data.

        Returns:
            np.ndarray: 1D array of original categorical values.

        Raises:
            ValueError: If the encoder has not been fitted or shape mismatch occurs.
        """
        if self.categories_ is None:
            raise ValueError("The encoder has not been fitted yet.")
        if one_hot.shape[1] != len(self.categories_):
            raise ValueError("The shape of the input does not match the number of categories.")
        return np.array([self.categories_[np.argmax(row)] for row in one_hot])


class ObjectiveScaler:
    """
    For label scaler in classification (binary and multiple classification)
    """
    def __init__(self, obj_name="sigmoid", ohe_scaler=None):
        """
        ohe_scaler: Need to be an instance of One-Hot-Encoder for softmax scaler (multiple classification problem)
        """
        self.obj_name = obj_name
        self.ohe_scaler = ohe_scaler

    def transform(self, data):
        if self.obj_name == "sigmoid" or self.obj_name == "self":
            return data
        elif self.obj_name == "hinge":
            data = np.squeeze(np.array(data))
            data[np.where(data == 0)] = -1
            return data
        elif self.obj_name == "softmax":
            data = self.ohe_scaler.transform(np.reshape(data, (-1, 1)))
            return data

    def inverse_transform(self, data):
        if self.obj_name == "sigmoid":
            data = np.squeeze(np.array(data))
            data = np.rint(data).astype(int)
        elif self.obj_name == "hinge":
            data = np.squeeze(np.array(data))
            data = np.ceil(data).astype(int)
            data[np.where(data == -1)] = 0
        elif self.obj_name == "softmax":
            data = np.squeeze(np.array(data))
            data = np.argmax(data, axis=1)
        return data


class Log1pScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # LogETransformer doesn't require fitting, so we simply return self.
        return self

    def transform(self, X):
        # Apply the natural logarithm to each element of the input data
        return np.log1p(X)

    def inverse_transform(self, X):
        # Apply the exponential function to reverse the logarithmic transformation
        return np.expm1(X)


class LogeScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # LogETransformer doesn't require fitting, so we simply return self.
        return self

    def transform(self, X):
        # Apply the natural logarithm (base e) to each element of the input data
        return np.log(X)

    def inverse_transform(self, X):
        # Apply the exponential function to reverse the logarithmic transformation
        return np.exp(X)


class SqrtScaler(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # SqrtScaler doesn't require fitting, so we simply return self.
        return self

    def transform(self, X):
        # Apply the square root transformation to each element of the input data
        return np.sqrt(X)

    def inverse_transform(self, X):
        # Apply the square of each element to reverse the square root transformation
        return X ** 2


class BoxCoxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, lmbda=None):
        self.lmbda = lmbda

    def fit(self, X, y=None):
        # Estimate the lambda parameter from the data if not provided
        if self.lmbda is None:
            _, self.lmbda = boxcox(X.flatten())
        return self

    def transform(self, X):
        # Apply the Box-Cox transformation to the data
        X_new = boxcox(X.flatten(), lmbda=self.lmbda)
        return X_new.reshape(X.shape)

    def inverse_transform(self, X):
        # Inverse transform using the original lambda parameter
        return inv_boxcox(X, self.lmbda)


class YeoJohnsonScaler(BaseEstimator, TransformerMixin):

    def __init__(self, lmbda=None):
        self.lmbda = lmbda

    def fit(self, X, y=None):
        # Estimate the lambda parameter from the data if not provided
        if self.lmbda is None:
            _, self.lmbda = yeojohnson(X.flatten())
        return self

    def transform(self, X):
        # Apply the Yeo-Johnson transformation to the data
        X_new = boxcox(X.flatten(), lmbda=self.lmbda)
        return X_new.reshape(X.shape)

    def inverse_transform(self, X):
        # Inverse transform using the original lambda parameter
        return inv_boxcox(X, self.lmbda)


class SinhArcSinhScaler(BaseEstimator, TransformerMixin):
    # https://stats.stackexchange.com/questions/43482/transformation-to-increase-kurtosis-and-skewness-of-normal-r-v
    def __init__(self, epsilon=0.1, delta=1.0):
        self.epsilon = epsilon
        self.delta = delta

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.sinh(self.delta * np.arcsinh(X) - self.epsilon)

    def inverse_transform(self, X):
        return np.sinh((np.arcsinh(X) + self.epsilon) / self.delta)


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    The class is used to transform data using different scaling techniques.

    Parameters
    ----------
    scaling_methods : str, tuple, list, or np.ndarray
        The name of the scaler you want to use. Supported scaler names are: 'standard', 'minmax', 'max-abs',
        'log1p', 'loge', 'sqrt', 'sinh-arc-sinh', 'robust', 'box-cox', 'yeo-johnson'.

    list_dict_paras : dict or list of dict
        The parameters for the scaler. If you have only one scaler, please use a dict. Otherwise, please use a list of dict.
    """

    SUPPORTED_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "max-abs": MaxAbsScaler,
                         "log1p": Log1pScaler, "loge": LogeScaler, "sqrt": SqrtScaler,
                         "sinh-arc-sinh": SinhArcSinhScaler, "robust": RobustScaler,
                         "box-cox": BoxCoxScaler, "yeo-johnson": YeoJohnsonScaler}

    def __init__(self, scaling_methods=('standard', ), list_dict_paras=None):
        """
        Initialize the DataTransformer.

        Parameters
        ----------
        scaling_methods : str or list/tuple of str
            One or more scaling methods to apply in sequence.
            Must be keys in SUPPORTED_SCALERS.

        list_dict_paras : dict or list of dict, optional
            Parameters for each scaler. If only one method is provided,
            a single dict is expected. If multiple methods are provided,
            a list of parameter dictionaries should be given.
        """
        if isinstance(scaling_methods, str):
            if list_dict_paras is None:
                self.list_dict_paras = [{}]
            elif isinstance(list_dict_paras, dict):
                self.list_dict_paras = [list_dict_paras]
            else:
                raise TypeError("Expected a single dict for list_dict_paras when using one scaling method.")
            self.scaling_methods = [scaling_methods]
        elif isinstance(scaling_methods, (list, tuple, np.ndarray)):
            if list_dict_paras is None:
                self.list_dict_paras = [{} for _ in range(len(scaling_methods))]
            elif isinstance(list_dict_paras, (list, tuple, np.ndarray)):
                self.list_dict_paras = list(list_dict_paras)
            else:
                raise TypeError("list_dict_paras should be a list/tuple of dicts when using multiple scaling methods.")
            self.scaling_methods = list(scaling_methods)
        else:
            raise TypeError("scaling_methods must be a str, list, tuple, or np.ndarray")

        self.scalers = [self._get_scaler(technique, paras) for (technique, paras) in
                        zip(self.scaling_methods, self.list_dict_paras)]

    @staticmethod
    def _ensure_2d(X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # convert (n,) to (n, 1)
        elif X.ndim != 2:
            raise ValueError(f"Input X must be 1D or 2D, but got shape {X.shape}")
        return X

    def _get_scaler(self, technique, paras):
        if technique in self.SUPPORTED_SCALERS.keys():
            if not isinstance(paras, dict):
                paras = {}
            return self.SUPPORTED_SCALERS[technique](**paras)
        else:
            raise ValueError(f"Unsupported scaling technique: '{technique}'. Supported techniques: {list(self.SUPPORTED_SCALERS)}")

    def fit(self, X, y=None):
        """
        Fit the sequence of scalers on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : Ignored
            Not used, exists for compatibility with sklearn's pipeline.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = self._ensure_2d(X)
        for idx, _ in enumerate(self.scalers):
            X = self.scalers[idx].fit_transform(X)
        return self

    def transform(self, X):
        """
        Transform the input data using the sequence of fitted scalers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : array-like
            Transformed data.
        """
        X = self._ensure_2d(X)
        for scaler in self.scalers:
            X = scaler.transform(X)
        return X

    def inverse_transform(self, X):
        """
        Reverse the transformations applied to the data.

        Parameters
        ----------
        X : array-like
            Transformed data to invert.

        Returns
        -------
        X_original : array-like
            Original data before transformation.
        """
        X = self._ensure_2d(X)
        for scaler in reversed(self.scalers):
            X = scaler.inverse_transform(X)
        return X
