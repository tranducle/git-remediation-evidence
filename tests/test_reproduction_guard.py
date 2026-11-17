"""Reject changed estimates, incomplete outputs and boolean/number coercion."""
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location('reproduction_guard', Path(__file__).resolve().parents[1] / 'scripts/reproduce.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_changed_estimate_fails_even_when_metadata_agree():
    expected = {'estimate': 0.125, 'counts': [13785, 74673], 'is_synthetic': False}
    actual = {'estimate': 0.126, 'counts': [13785, 74673], 'is_synthetic': False}
    assert replay.compare(expected, actual) == ['/estimate: numerical value differs']


def test_missing_result_and_shortened_vector_fail():
    assert replay.compare({'estimate': 0.1}, {})
    assert replay.compare([1, 2], [1])


def test_boolean_cannot_replace_estimate_or_vice_versa():
    assert replay.compare({'estimate': 0.0}, {'estimate': False})
    assert replay.compare({'is_synthetic': False}, {'is_synthetic': 0})
