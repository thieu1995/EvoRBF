#!/usr/bin/env python
# Created by "Thieu" at 23:33, 10/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from evorbf.helpers.scaler import DataTransformer
from sklearn.model_selection import train_test_split


class LabelEncoder:
    """
    Encode categorical labels as integer indices and decode them back.

    This class maps unique categorical labels to integers from 0 to n_classes - 1.
    """

    def __init__(self):
        """
        Initialize the label encoder.
        """
        self.unique_labels = None
        self.label_to_index = {}

    def fit(self, y):
        """
        Fit the encoder by finding unique labels in the input data.

        Parameters
        ----------
        y : array-like
            Input labels.

        Returns
        -------
        self : LabelEncoder
            Fitted LabelEncoder instance.
        """
        y = np.asarray(y).ravel()
        self.unique_labels = np.unique(y)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}
        return self

    def transform(self, y):
        """
        Transform labels to integer indices.

        Parameters
        ----------
        y : array-like
            Labels to encode.

        Returns
        -------
        encoded_labels : np.ndarray
            Encoded integer labels.

        Raises
        ------
        ValueError
            If the encoder has not been fitted or unknown labels are found.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = np.asarray(y).ravel()
        encoded = []
        for label in y:
            if label not in self.label_to_index:
                raise ValueError(f"Unknown label: {label}")
            encoded.append(self.label_to_index[label])
        return np.array(encoded)

    def fit_transform(self, y):
        """
        Fit the encoder and transform labels in one step.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Input labels.

        Returns
        -------
        np.ndarray
            Encoded integer labels.
        """
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        """
        Transform integer indices back to original labels.

        Parameters
        ----------
        y : array-like of int
            Encoded integer labels.

        Returns
        -------
        original_labels : np.ndarray
            Original labels.

        Raises
        ------
        ValueError
            If the encoder has not been fitted or index is out of bounds.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        y = np.asarray(y).ravel()
        return np.array([self.unique_labels[i] if 0 <= i < len(self.unique_labels) else "unknown" for i in y])


class TimeSeriesDifferencer:
    """
    A class for applying and reversing differencing on time series data.

    Differencing helps remove trends and seasonality from time series for better modeling.
    """

    def __init__(self, interval=1):
        """
        Initialize the differencer with a specified interval.

        Parameters
        ----------
        interval : int
            The lag interval to use for differencing. Must be >= 1.
        """
        if interval < 1:
            raise ValueError("Interval for differencing must be at least 1.")
        self.interval = interval
        self.original_data = None

    def difference(self, X):
        """
        Apply differencing to the input time series.

        Parameters
        ----------
        X : array-like
            The original time series data.

        Returns
        -------
        np.ndarray
            The differenced time series of length (len(X) - interval).
        """
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("Input must be a one-dimensional array.")
        self.original_data = X.copy()
        return np.array([X[i] - X[i - self.interval] for i in range(self.interval, len(X))])

    def inverse_difference(self, diff_data):
        """
        Reverse the differencing transformation using the stored original data.

        Parameters
        ----------
        diff_data : array-like
            The differenced data to invert.

        Returns
        -------
        np.ndarray
            The reconstructed original data (excluding the first `interval` values).

        Raises
        ------
        ValueError
            If the original data is not available.
        """
        if self.original_data is None:
            raise ValueError("Original data is required for inversion. Call difference() first.")
        diff_data = np.asarray(diff_data)
        return np.array([
            diff_data[i - self.interval] + self.original_data[i - self.interval]
            for i in range(self.interval, len(self.original_data))
        ])


class FeatureEngineering:
    """
    A class for performing custom feature engineering on numeric datasets.
    """

    def __init__(self):
        """
        Initialize the FeatureEngineering class.

        Currently, this class has no parameters but can be extended in the future.
        """
        pass

    def create_threshold_binary_features(self, X, threshold):
        """
        Add binary indicator columns to mark values below a given threshold.
        Each original column is followed by a new column indicating whether
        each value is below the threshold (1 if True, 0 otherwise).

        Parameters
        ----------
        X : numpy.ndarray
            The input 2D matrix of shape (n_samples, n_features).

        threshold : float
            The threshold value used to determine binary flags.

        Returns
        -------
        numpy.ndarray
            A new 2D matrix of shape (n_samples, 2 * n_features),
            where each original column is followed by its binary indicator column.

        Raises
        ------
        ValueError
            If `X` is not a NumPy array or not 2D.
            If `threshold` is not a numeric type.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X should be a NumPy array.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")
        if not isinstance(threshold, (int, float)):
            raise ValueError("Threshold should be a numeric value.")

        # Create a new matrix to hold original and new binary columns
        X_new = np.zeros((X.shape[0], X.shape[1] * 2), dtype=X.dtype)

        for idx in range(X.shape[1]):
            feature_values = X[:, idx]
            indicator_column = (feature_values < threshold).astype(int)
            X_new[:, idx * 2] = feature_values
            X_new[:, idx * 2 + 1] = indicator_column

        return X_new


class Data:
    """
    The structure of our supported Data class

    Parameters
    ----------
    X : np.ndarray
        The features of your data

    y : np.ndarray
        The labels of your data
    """

    SUPPORT = {
        "scaler": list(DataTransformer.SUPPORTED_SCALERS.keys())
    }

    def __init__(self, X=None, y=None, name="Unknown"):
        self.X = X
        self.y = self.check_y(y)
        self.name = name
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None

    @staticmethod
    def check_y(y):
        if y is None:
            return y
        y = np.squeeze(np.asarray(y))
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        return y

    @staticmethod
    def scale(X, scaling_methods=('standard', ), list_dict_paras=None):
        X = np.squeeze(np.asarray(X))
        if X.ndim == 1:
            X = np.reshape(X, (-1, 1))
        if X.ndim >= 3:
            raise TypeError(f"Invalid X data type. It should be array-like with shape (n samples, m features)")
        scaler = DataTransformer(scaling_methods=scaling_methods, list_dict_paras=list_dict_paras)
        data = scaler.fit_transform(X)
        return data, scaler

    @staticmethod
    def encode_label(y):
        y = np.squeeze(np.asarray(y))
        if y.ndim != 1:
            raise TypeError(f"Invalid y data type. It should be a vector / array-like with shape (n samples,)")
        scaler = LabelEncoder()
        data = scaler.fit_transform(y)
        return data, scaler

    def split_train_test(self, test_size=0.2, train_size=None,
                         random_state=41, shuffle=True, stratify=None, inplace=True):
        """
        The wrapper of the split_train_test function in scikit-learn library.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                        train_size=train_size, random_state=random_state, shuffle=shuffle, stratify=stratify)
        if not inplace:
            return self.X_train, self.X_test, self.y_train, self.y_test

    def set_train_test(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Function use to set your own X_train, y_train, X_test, y_test in case you don't want to use our split function

        Parameters
        ----------
        X_train : np.ndarray
        y_train : np.ndarray
        X_test : np.ndarray
        y_test : np.ndarray
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        return self
