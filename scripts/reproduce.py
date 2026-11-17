"""Recompute the empirical analyses and compare every numerical result with the reference."""
from pathlib import Path
import argparse
import importlib
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def compare(expected, actual, path='', differences=None):
    differences = differences if differences is not None else []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            differences.append(path + ': expected object')
            return differences
        for key, value in expected.items():
            if key not in actual:
                differences.append(path + '/' + key + ': missing')
            else:
                compare(value, actual[key], path + '/' + key, differences)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            differences.append(path + ': list length differs')
        else:
            for i, (left, right) in enumerate(zip(expected, actual)):
                compare(left, right, path + '/' + str(i), differences)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            differences.append(path + ': expected number')
        elif not (math.isnan(expected) and math.isnan(actual)) and not math.isclose(expected, actual, abs_tol=1e-10, rel_tol=1e-10):
            differences.append(path + ': numerical value differs')
    elif isinstance(expected, bool) and (not isinstance(actual, bool) or actual != expected):
        differences.append(path + ': boolean differs')
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', type=Path, default=ROOT / 'results/recomputed', help='Directory for regenerated results')
    args = parser.parse_args()
    output = args.o.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'DATA_MANIFEST.json').read_text())
    for entry in manifest['files']:
        target = ROOT / entry['path']
        if hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() != entry['sha256']:
            raise ValueError('Input/reference checksum differs: ' + entry['path'])
    measurement = importlib.import_module('measurement_analysis')
    measurement.COMPANION_DIR = output / 'measurement'
    measurement.main()
    activity = importlib.import_module('activity_analysis')
    activity.OUTPUT_PATH = output / 'activity_analysis.json'
    activity.main()
    comparisons = [
        (ROOT / 'results/reference/measurement_summary.json', output / 'measurement/measurement_summary.json'),
        (ROOT / 'results/reference/activity_analysis.json', output / 'activity_analysis.json'),
    ]
    differences = []
    for reference, computed in comparisons:
        differences.extend(compare(json.loads(reference.read_text()), json.loads(computed.read_text()), reference.name))
    receipt = {'status': 'PASS' if not differences else 'FAIL', 'numerical_tolerance': 1e-10, 'differences': differences}
    (output / 'reproduction.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0 if not differences else 1


if __name__ == '__main__':
    raise SystemExit(main())
