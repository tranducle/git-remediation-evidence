# Git remediation evidence

Code and research data for *Measuring Vulnerability Remediation Beyond Registry Eligibility: Git Evidence from npm, PyPI, and Maven*.

The study distinguishes an upstream release becoming eligible under a version range from a downstream repository committing a repair. It also examines repository activity, advisory curation and the balance between upgrading and dependency removal.

The released confirmatory frame contains 74,673 post-fix dependency exposures from 10,291 public GitHub repositories. Git state is recoverable for 13,785 exposures. Unknown histories remain unknown throughout the analyses.

## Reproduce the analyses

Use Python 3.14.5. Install the pinned environment, run the tests and regenerate results from the supplied data:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest tests/
python scripts/reproduce.py
```

The last command writes to `results/recomputed/` and compares numerical fields with `results/reference/`. Input files and reference results are left unchanged. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the scope of this replay and [DATA.md](DATA.md) for the data dictionary and source attribution.

## Repository contents

| Path | Contents |
| :- | :- |
| `src/` | Dependency parsers, estimator implementations, measurement comparisons and activity analyses |
| `data/` | Git-derived exposure records, covariates, signal panels and sampled workflow records |
| `results/` | Reference estimates, figure data and numerical analysis summaries |
| `configs/study.json` | Observation windows, fixed seeds, split parameters and inference budgets |
| `tests/` | Parser, version-grammar and estimator regression tests |

The results describe public direct dependencies in npm, PyPI and Maven. Repository-activity stratification is post-hoc. Missing Git histories and conditional recall assumptions limit generalization. The data contain candidate removals, not a claim that each candidate is a verified deletion.

Code is released under the MIT license. The authors' derived research data are available under CC BY 4.0; source attribution and retained third-party terms are described in [DATA.md](DATA.md).

## Citation

Use [CITATION.cff](CITATION.cff) to cite this software and data. The accompanying manuscript is unpublished; this repository does not assign it a journal publication or DOI.
