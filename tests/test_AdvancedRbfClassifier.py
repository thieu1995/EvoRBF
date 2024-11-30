#!/usr/bin/env python
# Created by "Thieu" at 00:52, 01/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
from sklearn.datasets import make_classification
from evorbf import AdvancedRbfClassifier


@pytest.fixture
def sample_data():
    """Fixture to provide a small dataset for testing."""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=3, random_state=42)
    return X, y


def test_fit(sample_data):
    """Test the fit method and ensure it sets n_labels correctly."""
    X, y = sample_data
    clf = AdvancedRbfClassifier()
    clf.fit(X, y)
    assert clf.n_labels is not None
    assert isinstance(clf.n_labels, int)


def test_fit_invalid_y():
    """Test that fit raises a TypeError with invalid y."""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=3, random_state=42)
    clf = AdvancedRbfClassifier()
    with pytest.raises(TypeError, match="Invalid y array type, it should be list, tuple or np.ndarray"):
        clf.fit(X, "invalid_label")


def test_score(sample_data):
    """Test the score method."""
    X, y = sample_data
    clf = AdvancedRbfClassifier()
    clf.fit(X, y)
    score = clf.score(X, y)
    assert isinstance(score, float)  # Assumes score should return a float (e.g., accuracy)


def test_scores(sample_data):
    """Test the scores method with a list of metrics."""
    X, y = sample_data
    clf = AdvancedRbfClassifier()
    clf.fit(X, y)
    metrics = clf.scores(X, y, list_metrics=("AS", "RS"))
    assert isinstance(metrics, dict)  # Assumes scores return a dictionary of metrics
    assert "AS" in metrics
    assert "RS" in metrics


def test_evaluate(sample_data):
    """Test the evaluate method."""
    X, y = sample_data
    clf = AdvancedRbfClassifier()
    clf.fit(X, y)
    y_pred = clf.network.predict(X)  # Assuming the network has a predict method
    metrics = clf.evaluate(y, y_pred, list_metrics=("AS", "RS"))
    assert isinstance(metrics, dict)  # Assumes evaluate returns a dictionary of metrics
    assert "AS" in metrics
    assert "RS" in metrics


def test_fit_invalid_y_shape():
    """Test that fit raises a TypeError when y has an invalid shape."""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=3, random_state=42)
    y_invalid = y.reshape((50, 2))  # Invalid shape for a classification target
    clf = AdvancedRbfClassifier()
    with pytest.raises(TypeError,
                       match="Invalid y array shape, it should be 1D vector containing labels 0, 1, 2,.. and so on."):
        clf.fit(X, y_invalid)
