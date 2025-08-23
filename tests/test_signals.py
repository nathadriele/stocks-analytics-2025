from src.strategy.generate_signals import (
    _to_signal_from_regression,
    _to_signal_from_classification,
)


def test_signal_from_regression_positive():
    assert _to_signal_from_regression(0.01, threshold=0.0) == 1
    assert _to_signal_from_regression(0.10, threshold=0.05) == 1


def test_signal_from_regression_negative():
    assert _to_signal_from_regression(-0.01, threshold=0.0) == -1
    assert _to_signal_from_regression(-0.10, threshold=-0.05) == -1


def test_signal_from_regression_flat():
    assert _to_signal_from_regression(0.0, threshold=0.0) == 0
    assert _to_signal_from_regression(0.05, threshold=0.05) == 0


def test_signal_from_classification_basic():
    assert _to_signal_from_classification(0.9, threshold=0.5) == 1
    assert _to_signal_from_classification(0.49, threshold=0.5) == -1
    assert _to_signal_from_classification(0.5, threshold=0.5) == -1
