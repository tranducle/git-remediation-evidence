# Reproducing the observed-data analyses

Run the commands in the README from a clean checkout using Python 3.14.5. The dependencies in `requirements.txt` match the environment used to test this public package. The original calculations used an 8-worker Apple Silicon workstation; the released analyses run locally from the included frames without network requests.

`python scripts/reproduce.py` recomputes measurement inflation, missingness bounds, event-time contrasts, activity-stratified estimates, bot contrasts, competing cumulative incidence and sensitivity analyses. It compares every numeric field in the generated analysis summaries with the reference summaries at absolute and relative tolerance `1e-10`. Booleans and list sizes must also agree. Filenames, code hashes and presentation strings are not numerical equivalence claims.

To write outside the checkout:

```sh
python scripts/reproduce.py -o /tmp/remediation-results
```

## What is preserved

The empirical input values, row ordering, exposure identifiers, model thresholds, random seeds, sampling probabilities and statistical formulas are preserved. Source files were renamed and their imports and paths changed to work within this repository. The parser and estimator regression tests accompany the public code.

The study identifier and allocation-salt strings are retained in code/configuration because they are inputs to deterministic seed derivation and sample allocation. They are scientific constants even where their historical spelling differs from the public repository name. Changing those strings changes the experiment.

## Replay boundary

This package replays the observed-data analyses from released derived frames. It does not repeat the collection of approximately 140 GB of third-party Git histories, reconstruct source blobs that upstream owners may remove, or distribute the private research administration used before data were unblinded. A new collection is not interchangeable with this recorded cohort.

The reusable parsing code operates on immutable snapshot dictionaries. To collect additional observations, retrieve the public repository histories identified in the exposure records, preserve commit timestamps and content hashes, and apply the same ecosystem grammars, observation windows and explicit-unknown rules. A complete new acquisition remains subject to the upstream providers' terms and repository licenses.

The public replay uses the supplied metadata projections instead of loading full third-party snapshot payloads. It retains the first constraint observation per anchor and the original ordering used by the analyses. No endpoint states or unknown exposures are inferred from missing source files.

The administrative mode named `official` in inherited low-level APIs is intentionally unavailable. It requires the original research administration and does not substitute a successful public replay for a new authorization to collect data. Use the documented reproduction command for this dataset.

## Known-process Monte Carlo calibration

The original calibration covers eight scenarios with 1000 replicates each. `configs/calibration.json` retains the exact base seed, scenario order, cluster counts and event times. `results/calibration/` retains the original numeric replicate records and diagnostics. The aggregate reference summary omits author administrative flags and path-dependent code metadata; its statistical values and validation scope are preserved.

Use the two calibration commands in the README. The output directory must not already exist. `scripts/verify_calibration.py` compares the complete 8000-replicate result and both diagnostic CSVs at absolute and relative tolerance 1e-10. The known processes do not establish empirical identification or validate fitted doubly robust nuisance models. The tested dependency closure is in requirements.lock; the calibration's direct library versions are also recorded in configs/environment.json.
