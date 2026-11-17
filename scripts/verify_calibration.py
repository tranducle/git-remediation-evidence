"""Compare a complete calibration replay with the released reference results."""
import json
from pathlib import Path
import sys
import pandas as pd
from reproduce import compare

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = Path(sys.argv[1]).resolve()
    reference = ROOT / 'results/calibration'
    expected = json.loads((reference / 'summary.json').read_text())
    actual = json.loads((output / 'summary.json').read_text())
    differences = compare(expected, actual)
    if actual.get('validation_status') != 'PASS' or actual.get('retained_replicates') != 8000:
        differences.append('A complete successful 8000-replicate calibration is required.')
    for name in ['replicates.csv', 'event_time_and_cause_diagnostics.csv']:
        try:
            pd.testing.assert_frame_equal(pd.read_csv(reference / name, low_memory=False), pd.read_csv(output / name, low_memory=False), check_exact=False, atol=1e-10, rtol=1e-10)
        except AssertionError as exc:
            differences.append(name + ': ' + str(exc))
    report = {'status': 'PASS' if not differences else 'FAIL', 'differences': differences, 'replicates': actual.get('retained_replicates')}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if not differences else 1


if __name__ == '__main__':
    raise SystemExit(main())
