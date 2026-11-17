# Research data

The authors collected public repository and vulnerability metadata and constructed the analysis frames provided here. Collection does not transfer ownership of third-party source code. This release contains derived observations and analysis inputs; it does not mirror the studied repositories, their source blobs, private account credentials or complete upstream vulnerability databases.

## Files and units

| File | Unit and purpose |
| :- | :- |
| `git_observations.parquet` | Exposure endpoint state, observability, Git identifiers and dependency scope |
| `first_exit.parquet` | Exposure-level first event, competing cause and censoring |
| `covariates.parquet` | Exposure-level pretreatment repository and dependency covariates |
| `measurement_variants.json` | Paired outcomes under alternative measurement definitions |
| `robustness_panel.json` | Exposure-level outcomes and covariates for sensitivity specifications |
| `signal_panels/*.npz` | Named NumPy arrays for baseline, EPSS, advisory, KEV and competing-risk analyses |
| `exposure_metadata.jsonl` | Anchor identifier, baseline missingness reason and signal timestamp, in original exposure order |
| `baseline_constraints.csv` | First encountered manifest-constraint evidence per anchor, in original encounter order |
| `workflow/sample.json` | Sample identifiers, strata and design weights for 120 workflow observations |
| `workflow/records/` | Timestamp-derived workflow classifications and trace flags |

The first three tables contain 74,673 rows in the same order. The two projected metadata files retain the fields consumed by the analyses. Collection-error paths are replaced with a stable error category in metadata and summary labels, with counts unchanged. Keeping the first constraint row per anchor matches the original analysis selection; it is not a new sample or outcome filter. Source hashes for the original records and the released projections are recorded in the data manifest.

The five states are vulnerable stasis, compatible manifest upgrade, breaking manifest upgrade, lockfile-only refresh and verified direct removal. Unrecoverable histories are separate from these five states. Exposures, anchors, CVEs, commit hashes, sample identifiers and fixed scientific version identifiers are retained. Public project paths refer to paths inside studied repositories.

## Sources and attribution

- [GitHub Advisory Database](https://github.com/github/advisory-database) and [PyPI advisories through OSV](https://google.github.io/osv.dev/data/): the relevant advisory sources use CC BY 4.0. OSV aggregates databases with different licenses; this release does not redistribute the full OSV corpus or assume a common license for every source.
- [FIRST EPSS](https://www.first.org/epss/): used to construct score-cutover and threshold indicators. Full daily score archives and proprietary exploitation telemetry are not mirrored here.
- [CISA KEV](https://github.com/cisagov/kev-data): the provider's catalog is distributed under CC0 1.0. Links to third-party material retain their own terms.
- [deps.dev](https://docs.deps.dev/api/v3/): used for package-release and dependency-resolution metadata. Provider-generated data use CC BY 4.0; aggregated source data retain their source terms.
- Public GitHub repositories: commit identifiers, manifest constraints and derived state transitions are used as observations. Repository contents and contributor copyrights remain with their owners.

The authors' derived tables and analysis frames are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute the authors and this repository, retain the source notices above and identify changes to the data. No ownership or endorsement by upstream providers is claimed.

`DATA_MANIFEST.json` enumerates the authorized release files with SHA-256 hashes, license and provenance. `RELEASE_SHA256.txt` covers the complete public artifact. NumPy archives are read with object unpickling disabled.
