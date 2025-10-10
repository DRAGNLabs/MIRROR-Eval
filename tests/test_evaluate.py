"""Tests for the evaluate module."""
import pytest
from mirror_eval import evaluate


def test_evaluate_returns_dict():
    """Test that evaluate function returns a dictionary."""
    result = evaluate()
    assert isinstance(result, dict)


def test_evaluate_has_status():
    """Test that evaluate result contains status field."""
    result = evaluate()
    assert "status" in result
    assert result["status"] == "success"


def test_evaluate_with_args():
    """Test that evaluate function accepts arguments."""
    result = evaluate("arg1", "arg2", key1="value1", key2="value2")
    assert isinstance(result, dict)
    assert result["status"] == "success"


def test_evaluate_has_version():
    """Test that evaluate result contains version information."""
    result = evaluate()
    assert "version" in result
    assert result["version"] == "0.1.0"
