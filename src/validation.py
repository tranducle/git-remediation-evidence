"""Keep synthetic calculations separate from the released empirical frames."""
from pathlib import Path


class EmpiricalGateBlockedError(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def assert_official_empirical_unlocked(*args, **kwargs):
    raise EmpiricalGateBlockedError(
        'New data collection is outside the published-data replay. '
        'Use scripts/reproduce.py to analyze the released observations.'
    )


def assert_synthetic_mode_isolated(input_paths=None, **kwargs):
    data_root = Path(__file__).resolve().parent.parent / 'data'
    for path in input_paths or []:
        if Path(path).resolve().is_relative_to(data_root):
            raise EmpiricalGateBlockedError('Synthetic runs cannot consume empirical inputs.')
    return {'isolated': True}
