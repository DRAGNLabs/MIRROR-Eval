"""Tests for the evaluate module."""
import inspect

from mirroreval import evaluate


def test_evaluate_is_importable():
    """Test that evaluate function is importable from the package."""
    assert callable(evaluate)


def test_evaluate_signature():
    """Test that evaluate accepts a settings_file_path argument."""
    sig = inspect.signature(evaluate)
    params = list(sig.parameters.keys())
    assert params == ["settings_file_path"]
