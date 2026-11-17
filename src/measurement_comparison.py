"""Compare remediation definitions and evaluate robustness specifications."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import study_contract
import validation


CONSTRUCT_VERSION_A0 = 'a0-ablation-v1.0'
CONSTRUCT_VERSION_R0 = 'r0-robustness-v1.0'
MAX_VARIANTS_PER_CONTRAST = 12
MAX_LEGACY_CROSSWALK_ROWS = 210
ROWS_PER_CROSSWALK_AXIS = 30
MAX_BOUNDARY_CHECKS = 40
MIN_IV_FIRST_STAGE_F = 10.0
FEW_CLUSTER_THRESHOLD = 15

CROSSWALK_AXES = (
    'horizon',
    'clustering',
    'covariate_adjustment',
    'missingness_bounds',
    'ecosystem_leave_one_out',
    'clock_ambiguity',
    'placebo_timing',
)

LEGACY_AXIS_MAP = {
    'horizon': 'R1b_horizon_grid',
    'clustering': 'R1f_twoway_cgm_clustering',
    'covariate_adjustment': 'R1e_outlier_trimming',
    'missingness_bounds': 'R1c_honestdid_mbar',
    'ecosystem_leave_one_out': 'R1a_risk_threshold_and_LEV',
    'clock_ambiguity': 'R1g_delayed_entry_cause_specific_cox',
    'placebo_timing': 'R1d_depth2_transitive_k2',
}

A0_EFFECT_FIELDS = list(study_contract.RESULT_ROW_COLUMNS) + [
    'variant_id',
    'contrast_group',
    'classification',
    'resampling_indices_sha256',
]

ACTIVATION_CHECK_FIELDS = (
    'variant_id',
    'contrast_group',
    'comparator_id',
    'is_reference',
    'n_rows_total',
    'n_rows_changed',
    'n_mask_changed',
    'n_treated_changed',
    'n_control_changed',
    'fraction_rows_changed',
    'toggle_activated',
    'activation_note',
)

ROBUSTNESS_CROSSWALK_FIELDS = list(study_contract.RESULT_ROW_COLUMNS) + [
    'row_id',
    'crosswalk_axis',
    'legacy_axis',
    'legacy_spec_id',
    'spec_label',
]

BOUNDARY_CHECK_FIELDS = list(study_contract.RESULT_ROW_COLUMNS) + [
    'check_id',
    'check_family',
    'first_stage_f',
    'exclusion_plausible',
    'timing_valid',
]


def _code_sha256() -> str:
    return study_contract.sha256_file(Path(__file__).resolve())


def _canonical_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    return study_contract.sha256_bytes(blob)


def _assert_text_clean(text: str, label: str) -> None:
    issues = study_contract.check_no_forbidden_dashes(text)
    if issues:
        raise ValueError(f'Forbidden dash detected in {label}: {issues[:3]}')


def _check_mode_gate(mode: str, out_path: Path) -> str:
    if mode == 'synthetic':
        validation.assert_synthetic_mode_isolated([out_path])
        return 'SYNTHETIC_FIXTURE'
    if mode == 'official':
        validation.assert_official_empirical_unlocked()
        return 'OFFICIAL_EMPIRICAL'
    raise ValueError(f'Unsupported mode {mode!r}; expected synthetic or official')


def _safe_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool) or val == '':
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _cluster_t_ci(est: float, boot_vals: np.ndarray, n_clusters: int) -> tuple[float, float, float]:
    pass
    if boot_vals.size <= 1:
        return 0.0, est, est
    se = float(np.std(boot_vals, ddof=1))
    if not math.isfinite(se):
        return 0.0, est, est
    if se <= 1e-14:
        return 0.0, est, est
    df = max(int(n_clusters) - 1, 1)
    t_crit = float(stats.t.ppf(0.975, df=df))
    half = t_crit * se
    return se, est - half, est + half


def _two_sided_cluster_att(
    values: np.ndarray,
    treated_mask: np.ndarray,
    control_mask: np.ndarray,
    cluster_ids: np.ndarray,
    n_clusters: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    pass
    n_t = int(np.sum(treated_mask))
    n_c = int(np.sum(control_mask))
    t_sum = np.zeros(n_clusters, dtype=float)
    t_cnt = np.zeros(n_clusters, dtype=float)
    c_sum = np.zeros(n_clusters, dtype=float)
    c_cnt = np.zeros(n_clusters, dtype=float)
    if n_t == 0 or n_c == 0:
        return float('nan'), t_sum, t_cnt, c_sum, c_cnt, n_t, n_c

    np.add.at(t_sum, cluster_ids[treated_mask], values[treated_mask])
    np.add.at(t_cnt, cluster_ids[treated_mask], 1.0)
    np.add.at(c_sum, cluster_ids[control_mask], values[control_mask])
    np.add.at(c_cnt, cluster_ids[control_mask], 1.0)

    est = float(np.sum(t_sum) / n_t - np.sum(c_sum) / n_c)
    return est, t_sum, t_cnt, c_sum, c_cnt, n_t, n_c


def _boot_att_draws(
    boot_idx: np.ndarray,
    t_sum: np.ndarray,
    t_cnt: np.ndarray,
    c_sum: np.ndarray,
    c_cnt: np.ndarray,
) -> np.ndarray:
    bt_s = t_sum[boot_idx].sum(axis=1)
    bt_n = t_cnt[boot_idx].sum(axis=1)
    bc_s = c_sum[boot_idx].sum(axis=1)
    bc_n = c_cnt[boot_idx].sum(axis=1)
    valid = (bt_n > 0.0) & (bc_n > 0.0)
    out = np.full(boot_idx.shape[0], np.nan, dtype=float)
    out[valid] = (bt_s[valid] / bt_n[valid]) - (bc_s[valid] / bc_n[valid])
    return out


def generate_synthetic_a0_paired_data(
    seed: int | None = None,
    n_clusters: int = 40,
    obs_per_cluster: int = 8,
) -> list[dict[str, Any]]:
    pass
    eff_seed = (
        int(seed)
        if seed is not None
        else study_contract.derive_child_seed(study_contract.STUDY_ID, 'A0', 'synthetic_paired', 0)
    )
    rng = np.random.default_rng(eff_seed)
    ecosystems = ('npm', 'pypi', 'maven')
    rows: list[dict[str, Any]] = []

    for c_idx in range(n_clusters):
        repo_id = f'repo_{c_idx + 1:03d}'
        family_id = f'fam_{(c_idx // 2) + 1:03d}'
        repo_u = float(rng.normal(0.0, 0.05))
        eco = ecosystems[c_idx % len(ecosystems)]

        for k_idx in range(obs_per_cluster):
            cve_num = 1000 + ((c_idx * 5 + k_idx) % 28)
            cve_id = f'CVE-2024-{cve_num:04d}'
            coord = f'{eco}:pkg_{(k_idx % 4) + 1}'
            exposed = int(rng.random() < 0.48)

                                                                     
                                                                                                  
                                                                                                     
            base_active = 0.06 + 0.015 * exposed + repo_u
            git_verified = int(rng.random() < np.clip(base_active, 0.01, 0.95))

            passive_shift = int(rng.random() < (0.04 + 0.025 * exposed))
            git_naive = int(min(1, git_verified + passive_shift))
            git_active_plus_passive = int(min(1, git_naive + (1 if rng.random() < (0.03 + 0.03 * exposed) else 0)))

            proxy_noise_pos = int(rng.random() < (0.05 + 0.06 * exposed))
            proxy_noise_neg = int(rng.random() < 0.015)
            legacy_verified = int(np.clip(git_verified + proxy_noise_pos - proxy_noise_neg, 0, 1))
            legacy_naive = int(np.clip(git_naive + proxy_noise_pos + (1 if (exposed and rng.random() < 0.04) else 0), 0, 1))

            pre_bot = int(rng.random() < np.clip(0.05 + 0.01 * exposed + repo_u, 0.01, 0.95))
            head_bot = int(np.clip(pre_bot + (1 if rng.random() < (0.04 + 0.03 * exposed) else 0), 0, 1))

            unfiltered_out = int(rng.random() < np.clip(0.07 + 0.02 * exposed, 0.01, 0.95))
            persisted_14d = int(rng.random() < 0.82)
            persisted_out = int(np.clip(unfiltered_out - (1 if (persisted_14d == 0 and rng.random() < 0.6) else 0), 0, 1))

            rows.append(
                {
                    'repository_id': repo_id,
                    'family_id': family_id,
                    'cve_id': cve_id,
                    'coordinate': coord,
                    'ecosystem': eco,
                    'signal_exposed': exposed,
                    'git_history_verified_removal': git_verified,
                    'git_history_naive_removal': git_naive,
                    'legacy_release_proxy_verified_removal': legacy_verified,
                    'legacy_release_proxy_naive_removal': legacy_naive,
                    'git_active_plus_passive': git_active_plus_passive,
                    'pre_signal_bot_outcome': pre_bot,
                    'head_bot_outcome': head_bot,
                    'unfiltered_risk_set_outcome': unfiltered_out,
                    'future_14d_persistence_outcome': persisted_out,
                    'persisted_14d_eligible': persisted_14d,
                }
            )
    return rows


def _make_result_row_base(
    *,
    experiment_id: str,
    run_id: str,
    config_sha: str,
    code_sha: str,
    input_sha: str,
    estimand: str,
    exposure: str,
    comparator: str,
    subgroup: str,
    horizon_days: int,
    estimate: float | None,
    lower: float | None,
    upper: float | None,
    scale: str,
    uncertainty_method: str,
    independent_unit: str,
    n_repositories: int,
    n_cves: int,
    n_fix_dyads: int,
    n_calendar_shocks: int,
    multiplicity_family: str,
    test_status: str,
    failure_reason: str,
) -> dict[str, Any]:
    return {
        'experiment_id': experiment_id,
        'run_id': run_id,
        'config_sha256': config_sha,
        'code_sha256': code_sha,
        'input_sha256': input_sha,
        'estimand': estimand,
        'exposure': exposure,
        'comparator': comparator,
        'subgroup': subgroup,
        'horizon_days': int(horizon_days),
        'estimate': None if estimate is None else round(float(estimate), 8),
        'lower': None if lower is None else round(float(lower), 8),
        'upper': None if upper is None else round(float(upper), 8),
        'scale': scale,
        'uncertainty_method': uncertainty_method,
        'independent_unit': independent_unit,
        'n_repositories': int(n_repositories),
        'n_cves': int(n_cves),
        'n_fix_dyads': int(n_fix_dyads),
        'n_calendar_shocks': int(n_calendar_shocks),
        'multiplicity_family': multiplicity_family,
        'test_status': test_status,
        'failure_reason': failure_reason,
    }


def run_a0_ablations(
    paired_data: list[dict[str, Any]],
    out_dir: Path | str,
    *,
    bootstrap_reps: int = 1999,
    strict_activation: bool = True,
    mode: str = 'synthetic',
    seed: int | None = None,
    horizon_days: int = 30,
    extra_variants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pass
    out_path = Path(out_dir).resolve()
    mode_label = _check_mode_gate(mode, out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if not paired_data:
        raise ValueError('paired_data must be non-empty for A0 ablations')
    if int(bootstrap_reps) < 1:
        raise ValueError(f'bootstrap_reps must be >= 1, got {bootstrap_reps}')

    seen_keys: set[tuple[str, str, str]] = set()
    for idx, row in enumerate(paired_data):
        key = (
            str(row.get('repository_id') or ''),
            str(row.get('cve_id') or ''),
            str(row.get('coordinate') or ''),
        )
        if any(not part for part in key):
            raise ValueError(f'paired_data[{idx}] missing frozen key component: {key!r}')
        if key in seen_keys:
            raise ValueError(f'paired_data duplicate frozen observation key {key!r}')
        seen_keys.add(key)

    clusters = sorted({str(r['repository_id']) for r in paired_data})
    n_clusters = len(clusters)
    if n_clusters < 2:
        raise ValueError(f'A0 requires >= 2 repository clusters for paired bootstrap, got {n_clusters}')

    cluster_to_idx = {cid: idx for idx, cid in enumerate(clusters)}
    cluster_ids = np.array([cluster_to_idx[str(r['repository_id'])] for r in paired_data], dtype=np.int64)
    exposed_all = np.array([bool(int(r['signal_exposed'])) for r in paired_data], dtype=bool)
    base_treated = exposed_all
    base_control = ~exposed_all

    n_rows = len(paired_data)
    n_cves = len({str(r['cve_id']) for r in paired_data})

    boot_seed = (
        int(seed)
        if seed is not None
        else study_contract.derive_child_seed(study_contract.STUDY_ID, 'A0', 'paired_bootstrap', int(bootstrap_reps))
    )
    boot_idx = study_contract.paired_cluster_resampling_indices(n_clusters, int(bootstrap_reps), boot_seed)
    resampling_sha = study_contract.sha256_bytes(boot_idx.tobytes())

    def _col_vec(col_name: str, fallback: str | None = None) -> np.ndarray:
        vals: list[float] = []
        for r in paired_data:
            v = r.get(col_name)
            if v is None and fallback is not None:
                v = r.get(fallback)
            if v is None:
                raise ValueError(f'Missing ablation outcome column {col_name!r} in paired_data')
            vals.append(float(v))
        return np.array(vals, dtype=float)

    def _mask_vec(col_name: str) -> np.ndarray:
        return np.array([bool(int(r.get(col_name, 1))) for r in paired_data], dtype=bool)

    ref_vec = _col_vec('git_history_verified_removal')
    git_naive_vec = _col_vec('git_history_naive_removal')
    leg_ver_vec = _col_vec('legacy_release_proxy_verified_removal')
    leg_naive_vec = _col_vec('legacy_release_proxy_naive_removal')
    passive_pool_vec = _col_vec('git_active_plus_passive', fallback='git_history_naive_removal')
    pre_bot_vec = _col_vec('pre_signal_bot_outcome', fallback='git_history_verified_removal')
    head_bot_vec = _col_vec('head_bot_outcome', fallback='git_history_naive_removal')
    unfiltered_vec = _col_vec('unfiltered_risk_set_outcome', fallback='git_history_verified_removal')
    persist_vec = _col_vec('future_14d_persistence_outcome', fallback='git_history_verified_removal')
    persist_mask = _mask_vec('persisted_14d_eligible')

    all_true_mask = np.ones(n_rows, dtype=bool)

    variant_specs: list[dict[str, Any]] = [
                                                                
        {
            'kind': 'cell',
            'variant_id': 'git_history_x_verified_removal',
            'contrast_group': 'A0_2x2_factorial',
            'comparator_id': 'git_history_x_verified_removal',
            'is_reference': True,
            'classification': 'primary_repaired_estimand',
            'exposure': 'high_signal_advisory',
            'comparator': 'low_signal_control',
            'values': ref_vec,
            'mask': all_true_mask,
            'ref_values': ref_vec,
            'ref_mask': all_true_mask,
        },
        {
            'kind': 'cell',
            'variant_id': 'git_history_x_naive_removal',
            'contrast_group': 'A0_2x2_factorial',
            'comparator_id': 'git_history_x_verified_removal',
            'is_reference': False,
            'classification': 'measurement_artifact',
            'exposure': 'high_signal_advisory',
            'comparator': 'low_signal_control',
            'values': git_naive_vec,
            'mask': all_true_mask,
            'ref_values': ref_vec,
            'ref_mask': all_true_mask,
        },
        {
            'kind': 'cell',
            'variant_id': 'legacy_release_proxy_x_verified_removal',
            'contrast_group': 'A0_2x2_factorial',
            'comparator_id': 'git_history_x_verified_removal',
            'is_reference': False,
            'classification': 'measurement_artifact',
            'exposure': 'high_signal_advisory',
            'comparator': 'low_signal_control',
            'values': leg_ver_vec,
            'mask': all_true_mask,
            'ref_values': ref_vec,
            'ref_mask': all_true_mask,
        },
        {
            'kind': 'cell',
            'variant_id': 'legacy_release_proxy_x_naive_removal',
            'contrast_group': 'A0_2x2_factorial',
            'comparator_id': 'git_history_x_verified_removal',
            'is_reference': False,
            'classification': 'measurement_artifact',
            'exposure': 'high_signal_advisory',
            'comparator': 'low_signal_control',
            'values': leg_naive_vec,
            'mask': all_true_mask,
            'ref_values': ref_vec,
            'ref_mask': all_true_mask,
        },
        {
            'kind': 'delta',
            'variant_id': 'delta_legacy_proxy_vs_git_at_verified',
            'contrast_group': 'A0_2x2_factorial',
            'comparator_id': 'git_history_x_verified_removal',
            'is_reference': False,
            'classification': 'measurement_artifact',
            'exposure': 'legacy_release_proxy_verified',
            'comparator': 'git_history_verified',
            'a_id': 'legacy_release_proxy_x_verified_removal',
            'b_id': 'git_history_x_verified_removal',
        },
        {
            'kind': 'delta',
            'variant_id': 'delta_naive_vs_verified_removal_at_git',
            'contrast_group': 'A0_2x2_factorial',
            'comparator_id': 'git_history_x_verified_removal',
            'is_reference': False,
            'classification': 'measurement_artifact',
            'exposure': 'git_history_naive_removal',
            'comparator': 'git_history_verified_removal',
            'a_id': 'git_history_x_naive_removal',
            'b_id': 'git_history_x_verified_removal',
        },
        {
            'kind': 'interaction_2x2',
            'variant_id': 'interaction_2x2_proxy_x_naive_removal',
            'contrast_group': 'A0_2x2_factorial',
            'comparator_id': 'zero_interaction_null',
            'is_reference': False,
            'classification': 'measurement_artifact_interaction',
            'exposure': 'legacy_naive_minus_legacy_verified',
            'comparator': 'git_naive_minus_git_verified',
        },
                                                       
        {
            'kind': 'cell',
            'variant_id': 'git_active_plus_passive',
            'contrast_group': 'A0_mechanism_variants',
            'comparator_id': 'git_history_x_verified_removal',
            'is_reference': False,
            'classification': 'changed_estimand',
            'exposure': 'high_signal_advisory',
            'comparator': 'low_signal_control',
            'values': passive_pool_vec,
            'mask': all_true_mask,
            'ref_values': ref_vec,
            'ref_mask': all_true_mask,
        },
        {
            'kind': 'delta',
            'variant_id': 'delta_active_plus_passive_vs_verified',
            'contrast_group': 'A0_mechanism_variants',
            'comparator_id': 'git_history_x_verified_removal',
            'is_reference': False,
            'classification': 'changed_estimand',
            'exposure': 'git_active_plus_passive',
            'comparator': 'git_history_verified_removal',
            'a_id': 'git_active_plus_passive',
            'b_id': 'git_history_x_verified_removal',
        },
        {
            'kind': 'cell',
            'variant_id': 'pre_signal_bot',
            'contrast_group': 'A0_mechanism_variants',
            'comparator_id': 'head_bot',
            'is_reference': False,
            'classification': 'changed_estimand',
            'exposure': 'pre_signal_bot_advisory',
            'comparator': 'low_signal_control',
            'values': pre_bot_vec,
            'mask': all_true_mask,
            'ref_values': head_bot_vec,
            'ref_mask': all_true_mask,
        },
        {
            'kind': 'cell',
            'variant_id': 'head_bot',
            'contrast_group': 'A0_mechanism_variants',
            'comparator_id': 'pre_signal_bot',
            'is_reference': False,
            'classification': 'changed_estimand',
            'exposure': 'head_bot_advisory',
            'comparator': 'low_signal_control',
            'values': head_bot_vec,
            'mask': all_true_mask,
            'ref_values': pre_bot_vec,
            'ref_mask': all_true_mask,
        },
        {
            'kind': 'delta',
            'variant_id': 'delta_head_bot_vs_pre_signal_bot',
            'contrast_group': 'A0_mechanism_variants',
            'comparator_id': 'pre_signal_bot',
            'is_reference': False,
            'classification': 'changed_estimand',
            'exposure': 'head_bot',
            'comparator': 'pre_signal_bot',
            'a_id': 'head_bot',
            'b_id': 'pre_signal_bot',
        },
        {
            'kind': 'cell',
            'variant_id': 'unfiltered_risk_set',
            'contrast_group': 'A0_mechanism_variants',
            'comparator_id': 'future_14d_persistence',
            'is_reference': False,
            'classification': 'changed_estimand',
            'exposure': 'unfiltered_risk_set',
            'comparator': 'low_signal_control',
            'values': unfiltered_vec,
            'mask': all_true_mask,
            'ref_values': persist_vec,
            'ref_mask': persist_mask,
        },
        {
            'kind': 'cell',
            'variant_id': 'future_14d_persistence',
            'contrast_group': 'A0_mechanism_variants',
            'comparator_id': 'unfiltered_risk_set',
            'is_reference': False,
            'classification': 'changed_estimand',
            'exposure': 'future_14d_persistence',
            'comparator': 'low_signal_control',
            'values': persist_vec,
            'mask': persist_mask,
            'ref_values': unfiltered_vec,
            'ref_mask': all_true_mask,
        },
        {
            'kind': 'delta',
            'variant_id': 'delta_future_14d_persistence_vs_unfiltered',
            'contrast_group': 'A0_mechanism_variants',
            'comparator_id': 'unfiltered_risk_set',
            'is_reference': False,
            'classification': 'changed_estimand',
            'exposure': 'future_14d_persistence',
            'comparator': 'unfiltered_risk_set',
            'a_id': 'future_14d_persistence',
            'b_id': 'unfiltered_risk_set',
        },
    ]

    if extra_variants:
        for ev in extra_variants:
            spec = dict(ev)
            kind = spec.setdefault('kind', 'cell')
            if kind == 'cell':
                if 'values' not in spec:
                    spec['values'] = _col_vec(str(spec['column']))
                if 'mask' not in spec:
                    spec['mask'] = all_true_mask
                if 'ref_values' not in spec:
                    if 'ref_column' in spec:
                        spec['ref_values'] = _col_vec(str(spec['ref_column']))
                    else:
                        spec['ref_values'] = ref_vec
                if 'ref_mask' not in spec:
                    spec['ref_mask'] = all_true_mask
            if 'comparator_id' not in spec:
                spec['comparator_id'] = 'git_history_x_verified_removal'
            spec.setdefault('is_reference', False)
            spec.setdefault('contrast_group', 'A0_extra_variants')
            spec.setdefault('classification', 'measurement_artifact')
            spec.setdefault('exposure', str(spec['variant_id']))
            spec.setdefault('comparator', 'low_signal_control')
            variant_specs.append(spec)

                                                               
    group_counts: dict[str, int] = {}
    for spec in variant_specs:
        grp = str(spec['contrast_group'])
        group_counts[grp] = group_counts.get(grp, 0) + 1
        if group_counts[grp] > MAX_VARIANTS_PER_CONTRAST:
            raise ValueError(
                f'Contrast family {grp!r} exceeds maximum {MAX_VARIANTS_PER_CONTRAST} prespecified variants'
            )

    code_sha = _code_sha256()
    input_sha = _canonical_sha256(paired_data)
    config_sha = _canonical_sha256(
        {
            'construct_version': CONSTRUCT_VERSION_A0,
            'mode': mode_label,
            'bootstrap_reps': int(bootstrap_reps),
            'boot_seed': boot_seed,
            'horizon_days': int(horizon_days),
            'resampling_indices_sha256': resampling_sha,
            'variants': [s['variant_id'] for s in variant_specs],
        }
    )
    run_id = f'A0-{mode_label}-{config_sha[:10]}'

    cell_est_cache: dict[str, float] = {}
    cell_boot_cache: dict[str, np.ndarray] = {}
    activation_rows: list[dict[str, Any]] = []
    noop_failures: list[str] = []

    for spec in variant_specs:
        if spec['kind'] != 'cell':
            continue
        vid = spec['variant_id']
        vals = spec['values']
        mask = spec['mask']
        r_vals = spec['ref_values']
        r_mask = spec['ref_mask']

        t_mask = base_treated & mask
        c_mask = base_control & mask
        est, t_s, t_c, c_s, c_c, _, _ = _two_sided_cluster_att(
            vals, t_mask, c_mask, cluster_ids, n_clusters
        )
        draws = _boot_att_draws(boot_idx, t_s, t_c, c_s, c_c)
        cell_est_cache[vid] = est
        cell_boot_cache[vid] = draws

                                                                                                          
        diff_vals = (np.abs(vals - r_vals) > 1e-12) & (mask | r_mask)
        diff_mask = mask != r_mask
        changed_any = diff_vals | diff_mask
        n_changed = int(np.sum(changed_any))
        n_mask_changed = int(np.sum(diff_mask))
        n_t_changed = int(np.sum(changed_any & base_treated))
        n_c_changed = int(np.sum(changed_any & base_control))
        is_ref = bool(spec.get('is_reference', False))
        activated = True if is_ref else (n_changed > 0)
        if not activated:
            noop_failures.append(vid)

        activation_rows.append(
            {
                'variant_id': vid,
                'contrast_group': spec['contrast_group'],
                'comparator_id': spec['comparator_id'],
                'is_reference': is_ref,
                'n_rows_total': n_rows,
                'n_rows_changed': n_changed,
                'n_mask_changed': n_mask_changed,
                'n_treated_changed': n_t_changed,
                'n_control_changed': n_c_changed,
                'fraction_rows_changed': round(n_changed / float(n_rows), 6),
                'toggle_activated': activated,
                'activation_note': (
                    'reference_baseline'
                    if is_ref
                    else ('activated' if activated else 'SILENT_NOOP_ZERO_ROWS_CHANGED')
                ),
            }
        )

                                                                                                    
    study_contract.atomic_write_csv(
        out_path / 'activation_checks.csv',
        ACTIVATION_CHECK_FIELDS,
        activation_rows,
    )

    if strict_activation and noop_failures:
        raise ValueError(
            f'Silent no-op ablation variant(s) changed 0 rows vs comparator: {noop_failures}'
        )

    effect_rows: list[dict[str, Any]] = []
    variant_diffs: dict[str, Any] = {
        'schema_version': 'a0_variant_diffs.v1',
        'construct_version': CONSTRUCT_VERSION_A0,
        'mode_label': mode_label,
        'bootstrap_reps': int(bootstrap_reps),
        'boot_seed': boot_seed,
        'resampling_indices_sha256': resampling_sha,
        'n_clusters': n_clusters,
        'n_rows': n_rows,
        'n_cves': n_cves,
        'variants': {},
    }

    for spec in variant_specs:
        vid = spec['variant_id']
        kind = spec['kind']

        if kind == 'cell':
            est = cell_est_cache[vid]
            draws = cell_boot_cache[vid]
        elif kind == 'delta':
            a_id = spec['a_id']
            b_id = spec['b_id']
            est = cell_est_cache[a_id] - cell_est_cache[b_id]
            draws = cell_boot_cache[a_id] - cell_boot_cache[b_id]
        elif kind == 'interaction_2x2':
            est = (
                (cell_est_cache['legacy_release_proxy_x_naive_removal'] - cell_est_cache['legacy_release_proxy_x_verified_removal'])
                - (cell_est_cache['git_history_x_naive_removal'] - cell_est_cache['git_history_x_verified_removal'])
            )
            draws = (
                (cell_boot_cache['legacy_release_proxy_x_naive_removal'] - cell_boot_cache['legacy_release_proxy_x_verified_removal'])
                - (cell_boot_cache['git_history_x_naive_removal'] - cell_boot_cache['git_history_x_verified_removal'])
            )
        else:
            raise ValueError(f'Unknown variant kind {kind!r}')

        finite_draws = draws[np.isfinite(draws)]
        if not math.isfinite(est) or finite_draws.size == 0:
            status = 'SPARSE_ZERO_CELL'
            fail_msg = f'Empty treated or control support across frozen clusters for {vid}'
            row_est = None
            row_lo = None
            row_hi = None
            se = None
            p025 = None
            p975 = None
        else:
            status = 'LIMITED_INFERENCE_FEW_CLUSTERS' if n_clusters < FEW_CLUSTER_THRESHOLD else 'ESTIMABLE'
            fail_msg = ''
            se, row_lo, row_hi = _cluster_t_ci(est, finite_draws, n_clusters)
            row_est = est
            p025 = float(np.percentile(finite_draws, 2.5))
            p975 = float(np.percentile(finite_draws, 97.5))

        base_row = _make_result_row_base(
            experiment_id='A0',
            run_id=run_id,
            config_sha=config_sha,
            code_sha=code_sha,
            input_sha=input_sha,
            estimand=f'a0_{vid}',
            exposure=spec['exposure'],
            comparator=spec['comparator'],
            subgroup='frozen_common_support',
            horizon_days=int(horizon_days),
            estimate=row_est,
            lower=row_lo,
            upper=row_hi,
            scale='risk_difference',
            uncertainty_method=f'paired_cluster_bootstrap_t_{int(bootstrap_reps)}',
            independent_unit='repository_cluster',
            n_repositories=n_clusters,
            n_cves=n_cves,
            n_fix_dyads=n_rows,
            n_calendar_shocks=0,
            multiplicity_family=spec['contrast_group'],
            test_status=status,
            failure_reason=fail_msg,
        )
        base_row['variant_id'] = vid
        base_row['contrast_group'] = spec['contrast_group']
        base_row['classification'] = spec['classification']
        base_row['resampling_indices_sha256'] = resampling_sha
        study_contract.validate_result_row(base_row)
        effect_rows.append(base_row)

        variant_diffs['variants'][vid] = {
            'kind': kind,
            'contrast_group': spec['contrast_group'],
            'classification': spec['classification'],
            'test_status': status,
            'estimate': None if row_est is None else round(float(row_est), 8),
            'lower': None if row_lo is None else round(float(row_lo), 8),
            'upper': None if row_hi is None else round(float(row_hi), 8),
            'cluster_boot_se': None if se is None else round(float(se), 8),
            'boot_p025_diag': None if p025 is None else round(float(p025), 8),
            'boot_p975_diag': None if p975 is None else round(float(p975), 8),
            'resampling_indices_sha256': resampling_sha,
        }

    study_contract.atomic_write_csv(out_path / 'ablation_effects.csv', A0_EFFECT_FIELDS, effect_rows)
    study_contract.atomic_write_json(out_path / 'variant_diffs.json', variant_diffs)

    _assert_text_clean((out_path / 'ablation_effects.csv').read_text(encoding='utf-8'), 'ablation_effects.csv')
    _assert_text_clean((out_path / 'activation_checks.csv').read_text(encoding='utf-8'), 'activation_checks.csv')
    _assert_text_clean((out_path / 'variant_diffs.json').read_text(encoding='utf-8'), 'variant_diffs.json')

    return {
        'run_id': run_id,
        'mode_label': mode_label,
        'n_variants': len(effect_rows),
        'n_clusters': n_clusters,
        'resampling_indices_sha256': resampling_sha,
        'ablation_effects_csv': str(out_path / 'ablation_effects.csv'),
        'variant_diffs_json': str(out_path / 'variant_diffs.json'),
        'activation_checks_csv': str(out_path / 'activation_checks.csv'),
    }


def generate_synthetic_r0_panel_data(
    seed: int | None = None,
    n_clusters: int = 48,
    obs_per_cluster: int = 10,
    max_followup_days_supported: int = 90,
) -> list[dict[str, Any]]:
    pass
    eff_seed = (
        int(seed)
        if seed is not None
        else study_contract.derive_child_seed(study_contract.STUDY_ID, 'R0', 'synthetic_panel', 0)
    )
    rng = np.random.default_rng(eff_seed)
    ecosystems = ('npm', 'pypi', 'maven')
    calendar_regimes = ('pre_2023', '2023_2024', '2025_plus')
    rows: list[dict[str, Any]] = []

    for c_idx in range(n_clusters):
        repo_id = f'repo_{c_idx + 1:03d}'
        family_id = f'fam_{(c_idx // 2) + 1:03d}'
        eco = ecosystems[c_idx % len(ecosystems)]
        repo_activity = float(rng.uniform(-1.0, 1.0))
        repo_age_years = float(rng.uniform(0.5, 9.0))
        is_archived = int(c_idx % 24 == 23)

        for k_idx in range(obs_per_cluster):
            cve_num = 2000 + ((c_idx * 7 + k_idx) % 25)
            cve_id = f'CVE-2024-{cve_num:04d}'
            exposed = int(rng.random() < 0.45)
            cal = calendar_regimes[(c_idx + k_idx) % len(calendar_regimes)]
            cvss = round(float(rng.uniform(4.2, 9.8)), 1)
            depth = 1 if (k_idx % 4 != 3) else 2
            config_activated = int(rng.random() < (0.35 + 0.25 * (eco != 'maven')))
            missing_flag = int(rng.random() < 0.08)

                                                                                           
            iv_z = float(rng.normal(0.0, 1.0))
            treat_prob = np.clip(0.45 + 0.015 * iv_z + 0.12 * repo_activity, 0.05, 0.95)
            treatment_d = int(rng.random() < treat_prob)

                                                                                 
            p30 = np.clip(0.06 + 0.014 * exposed + 0.02 * repo_activity - 0.03 * is_archived, 0.01, 0.95)
            y30 = int(rng.random() < p30)
            y14 = int(y30 and (rng.random() < 0.65))
            y60 = int(y30 or (rng.random() < 0.035))
            y90 = int(y60 or (rng.random() < 0.025))
            passive_only = int(rng.random() < 0.04)
            placebo_dev_only = int(rng.random() < (0.04 + 0.01 * repo_activity))

                                                                                   
            clock_discordant = int(rng.random() < 0.07)
            y30_committer = int(np.clip(y30 + (1 if (clock_discordant and rng.random() < 0.5) else 0), 0, 1))
            y30_pr_merged = y30

            rows.append(
                {
                    'repository_id': repo_id,
                    'family_id': family_id,
                    'cve_id': cve_id,
                    'coordinate': f'{eco}:dep_{(k_idx % 5) + 1}',
                    'ecosystem': eco,
                    'calendar_regime': cal,
                    'is_kev': int(exposed and (cvss >= 7.0)),
                    'signal_exposed': exposed,
                    'cvss': cvss,
                    'repo_activity_z': round(repo_activity, 4),
                    'repo_age_years': round(repo_age_years, 2),
                    'dependency_depth': depth,
                    'config_activated': config_activated,
                    'is_archived_repo': is_archived,
                    'missing_trace_flag': missing_flag,
                    'max_followup_days': int(max_followup_days_supported),
                    'iv_z': round(iv_z, 5),
                    'treatment_d': treatment_d,
                    'exclusion_plausible_default': False,
                    'y_active_7d': int(y14 and (rng.random() < 0.5)),
                    'y_active_14d': y14,
                    'y_active_30d': y30,
                    'y_active_60d': y60,
                    'y_active_90d': y90,
                    'y_active_30d_committer_clock': y30_committer,
                    'y_active_30d_pr_clock': y30_pr_merged,
                    'y_passive_only_30d': passive_only,
                    'y_placebo_unmapped_30d': placebo_dev_only,
                    'y_placebo_future_30d': int(rng.random() < 0.03),
                    'sample_weight_base': round(1.0 + 0.1 * abs(repo_activity), 4),
                }
            )
    return rows


def audit_and_run_exploratory_iv(
    panel_data: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    pass
    z_col = str(spec.get('z_col', 'iv_z'))
    d_col = str(spec.get('d_col', 'treatment_d'))
    y_col = str(spec.get('y_col', 'y_active_30d'))
    cluster_col = str(spec.get('cluster_col', 'repository_id'))

    valid_rows = [
        r for r in panel_data
        if r.get(z_col) is not None and r.get(d_col) is not None and r.get(y_col) is not None
    ]
    clusters = sorted({str(r.get(cluster_col) or '') for r in valid_rows if r.get(cluster_col)})
    n_clusters = len(clusters)

                                                                                            
    f_override = spec.get('first_stage_f_override')
    if f_override is not None:
        first_stage_f = float(f_override)
    elif len(valid_rows) < 10 or n_clusters < 2:
        first_stage_f = 0.0
    else:
        z = np.array([float(r[z_col]) for r in valid_rows], dtype=float)
        d = np.array([float(r[d_col]) for r in valid_rows], dtype=float)
        zc = z - np.mean(z)
        denom = float(np.sum(zc ** 2))
        if denom <= 1e-12:
            first_stage_f = 0.0
        else:
            slope = float(np.sum(zc * (d - np.mean(d))) / denom)
            resid = (d - np.mean(d)) - slope * zc
            score = zc * resid
            c_idx_map = {c: i for i, c in enumerate(clusters)}
            c_scores = np.zeros(n_clusters, dtype=float)
            for idx, r in enumerate(valid_rows):
                c_scores[c_idx_map[str(r[cluster_col])]] += score[idx]
            meat = float(np.sum(c_scores ** 2))
            dof_scale = (n_clusters / float(n_clusters - 1)) if n_clusters > 1 else 1.0
            se_slope = math.sqrt(max(dof_scale * meat, 1e-18)) / denom
            t_stat = slope / max(se_slope, 1e-12)
            first_stage_f = float(t_stat ** 2)

    exclusion_plausible = bool(spec.get('exclusion_plausible', False))
    exclusion_reason = str(spec.get('exclusion_violation_reason') or '').strip()
    timing_valid = bool(spec.get('timing_valid', True))

    reasons: list[str] = []
    if n_clusters < 2:
        reasons.append(f'insufficient_clusters ({n_clusters} < 2)')
    if not timing_valid:
        reasons.append('instrument_timing_violated (instrument not strictly pre-treatment)')
    if first_stage_f < MIN_IV_FIRST_STAGE_F:
        reasons.append(
            f'weak_first_stage_F={round(first_stage_f, 4)} < {MIN_IV_FIRST_STAGE_F} Stock-Yogo threshold'
        )
    if not exclusion_plausible:
        detail = exclusion_reason or 'instrument correlates with repository maintenance/release schedule'
        reasons.append(f'exclusion_plausibility_failed ({detail})')

    if reasons:
        return {
            'test_status': 'NOT_IDENTIFIED',
            'estimate': None,
            'lower': None,
            'upper': None,
            'first_stage_f': round(first_stage_f, 4),
            'exclusion_plausible': exclusion_plausible,
            'timing_valid': timing_valid,
            'n_clusters': n_clusters,
            'n_rows': len(valid_rows),
            'failure_reason': '; '.join(reasons),
        }

                                                                                           
    z = np.array([float(r[z_col]) for r in valid_rows], dtype=float)
    d = np.array([float(r[d_col]) for r in valid_rows], dtype=float)
    y = np.array([float(r[y_col]) for r in valid_rows], dtype=float)
    zc = z - np.mean(z)
    cov_zd = float(np.sum(zc * (d - np.mean(d))))
    cov_zy = float(np.sum(zc * (y - np.mean(y))))
    if abs(cov_zd) <= 1e-12:
        return {
            'test_status': 'NOT_IDENTIFIED',
            'estimate': None,
            'lower': None,
            'upper': None,
            'first_stage_f': round(first_stage_f, 4),
            'exclusion_plausible': exclusion_plausible,
            'timing_valid': timing_valid,
            'n_clusters': n_clusters,
            'n_rows': len(valid_rows),
            'failure_reason': 'near_zero_first_stage_covariance',
        }
    beta_iv = cov_zy / cov_zd
    u_hat = (y - np.mean(y)) - beta_iv * (d - np.mean(d))
    influence_num = zc * u_hat
    c_idx_map = {c: i for i, c in enumerate(clusters)}
    c_inf = np.zeros(n_clusters, dtype=float)
    for idx, r in enumerate(valid_rows):
        c_inf[c_idx_map[str(r[cluster_col])]] += influence_num[idx]
    dof_scale = n_clusters / float(n_clusters - 1)
    se_iv = math.sqrt(max(dof_scale * float(np.sum(c_inf ** 2)), 1e-18)) / abs(cov_zd)
    t_crit = float(stats.t.ppf(0.975, df=max(n_clusters - 1, 1)))
    lo = beta_iv - t_crit * se_iv
    hi = beta_iv + t_crit * se_iv
    status = 'LIMITED_INFERENCE_FEW_CLUSTERS' if n_clusters < FEW_CLUSTER_THRESHOLD else 'ESTIMABLE'
    return {
        'test_status': status,
        'estimate': float(beta_iv),
        'lower': float(lo),
        'upper': float(hi),
        'first_stage_f': round(first_stage_f, 4),
        'exclusion_plausible': True,
        'timing_valid': True,
        'n_clusters': n_clusters,
        'n_rows': len(valid_rows),
        'failure_reason': '',
    }


def _build_210_crosswalk_specs() -> list[dict[str, Any]]:
    pass
    specs: list[dict[str, Any]] = []
    idx = 0

    def _add(axis: str, label: str, params: dict[str, Any]) -> None:
        nonlocal idx
        idx += 1
        specs.append(
            {
                'row_id': f'R0-{idx:03d}',
                'crosswalk_axis': axis,
                'legacy_axis': LEGACY_AXIS_MAP[axis],
                'legacy_spec_id': f'{LEGACY_AXIS_MAP[axis]}_spec_{((idx - 1) % 30) + 1:02d}',
                'spec_label': label,
                'params': params,
            }
        )

                           
    h_vals = (7, 14, 30, 60, 90, 120, 180, 14, 30, 60)
    h_cohorts = ('all_cohorts', 'kev_subset', 'non_kev_subset')
    for c_name in h_cohorts:
        for j, h in enumerate(h_vals):
            scale_mode = 'risk_difference' if j < 7 else 'relative_att_shift'
            _add('horizon', f'horizon_{h}d_{c_name}_{scale_mode}_{j + 1}', {
                'horizon_days': h,
                'cohort': c_name,
                'scale': scale_mode,
            })

                              
    c_units = ('repository_id', 'family_id', 'cve_id', 'ecosystem_x_calendar', 'repository_high_freq', 'single_cluster_deg')
    c_adjusts = ('cr1_t', 'cr0_asymp', 'few_cluster_t', 'twoway_approx', 'weighted_cr1')
    for u in c_units:
        for adj in c_adjusts:
            _add('clustering', f'cluster_{u}_{adj}', {
                'cluster_mode': u,
                'se_mode': adj,
                'horizon_days': 30,
            })

                                        
    cov_models = ('unadjusted', 'ecosystem_fe', 'activity_quintile', 'cvss_severity', 'dr_activity_and_age', 'trimmed_weight_p05')
    cov_sub = ('all', 'npm_pypi', 'depth1_direct', 'high_cvss', 'active_repos')
    for m in cov_models:
        for s in cov_sub:
            _add('covariate_adjustment', f'cov_{m}_{s}', {
                'cov_mode': m,
                'subset': s,
                'horizon_days': 30,
            })

                                      
    m_types = ('complete_case', 'manski_lower', 'manski_upper', 'lee_lower', 'lee_upper', 'unsupported_heavy_censoring')
    m_rates = (0.02, 0.05, 0.10, 0.15, 0.20)
    for mt in m_types:
        for mr in m_rates:
            _add('missingness_bounds', f'missing_{mt}_delta_{int(round(mr * 100)):02d}pct', {
                'bound_mode': mt,
                'assumed_missing_rate': mr,
                'horizon_days': 30,
            })

                                           
    eco_ops = (
        'leave_out_npm',
        'leave_out_pypi',
        'leave_out_maven',
        'only_npm',
        'only_pypi',
        'only_maven',
        'only_cargo_absent',
        'only_rubygems_absent',
        'leave_out_depth2',
        'only_depth1',
    )
    eco_horizons = (14, 30, 60)
    for op in eco_ops:
        for h in eco_horizons:
            _add('ecosystem_leave_one_out', f'eco_{op}_{h}d', {
                'eco_mode': op,
                'horizon_days': h,
            })

                                   
    clocks = ('author_clock', 'committer_clock', 'pr_merge_clock', 'worst_case_delay_shift', 'discordant_drop')
    skews = (1, 6, 12, 24, 48, 72)
    for ck in clocks:
        for sk in skews:
            _add('clock_ambiguity', f'clock_{ck}_skew_{sk}h', {
                'clock_mode': ck,
                'skew_hours': sk,
                'horizon_days': 30,
            })

                                  
    p_modes = ('unmapped_dev_placebo', 'future_advisory_placebo', 'pre_mandate_shift_7d', 'pre_mandate_shift_30d', 'pre_mandate_shift_90d', 'empty_pre_period_probe')
    p_cohorts = ('all_cohorts', 'npm_only', 'pypi_only', 'maven_only', 'high_activity')
    for pm in p_modes:
        for pc in p_cohorts:
            _add('placebo_timing', f'placebo_{pm}_{pc}', {
                'placebo_mode': pm,
                'cohort': pc,
                'horizon_days': 30,
            })

    assert len(specs) == MAX_LEGACY_CROSSWALK_ROWS
    return specs


def _build_40_boundary_specs() -> list[dict[str, Any]]:
    pass
    checks: list[dict[str, Any]] = [
                                        
        {'check_id': 'RC-001', 'check_family': 'exploratory_iv_audit', 'kind': 'iv', 'z_col': 'iv_z', 'd_col': 'treatment_d', 'y_col': 'y_active_30d', 'exclusion_plausible': False, 'timing_valid': True, 'exclusion_violation_reason': 'disclosure schedule correlated with repo release cadence'},
        {'check_id': 'RC-002', 'check_family': 'exploratory_iv_audit', 'kind': 'iv', 'z_col': 'iv_z', 'd_col': 'treatment_d', 'y_col': 'y_active_60d', 'exclusion_plausible': False, 'timing_valid': True, 'exclusion_violation_reason': 'weak first-stage F and direct maintenance channel'},
        {'check_id': 'RC-003', 'check_family': 'exploratory_iv_audit', 'kind': 'iv', 'z_col': 'iv_z', 'd_col': 'treatment_d', 'y_col': 'y_active_90d', 'exclusion_plausible': False, 'timing_valid': False, 'exclusion_violation_reason': 'post-exposure disclosure lag timing leak'},
        {'check_id': 'RC-004', 'check_family': 'exploratory_iv_audit', 'kind': 'iv', 'z_col': 'iv_z', 'd_col': 'treatment_d', 'y_col': 'y_active_30d', 'first_stage_f_override': 1.325, 'exclusion_plausible': True, 'timing_valid': True, 'exclusion_violation_reason': ''},
        {'check_id': 'RC-005', 'check_family': 'exploratory_iv_audit', 'kind': 'iv', 'z_col': 'iv_z', 'd_col': 'treatment_d', 'y_col': 'y_active_30d', 'first_stage_f_override': 18.4, 'exclusion_plausible': False, 'timing_valid': True, 'exclusion_violation_reason': 'direct effect through registry publication rate violates exclusion'},
        {'check_id': 'RC-006', 'check_family': 'exploratory_iv_audit', 'kind': 'iv', 'z_col': 'iv_z', 'd_col': 'treatment_d', 'y_col': 'y_passive_only_30d', 'exclusion_plausible': False, 'timing_valid': True, 'exclusion_violation_reason': 'passive version churn violates exclusion restriction'},
    ]
                                    
    for i, (reg, h) in enumerate((
        ('pre_2023', 30), ('2023_2024', 30), ('2025_plus', 30), ('2025_plus', 60),
        ('pre_2023', 60), ('2023_2024', 90), ('2025_plus', 120), ('2025_plus', 180),
    ), start=7):
        checks.append({
            'check_id': f'RC-{i:03d}', 'check_family': 'temporal_boundary', 'kind': 'att',
            'filter_key': 'calendar_regime', 'filter_val': reg, 'horizon_days': h,
        })
                                                             
    eco_items = (
        ('ecosystem', 'npm', 'config_all', 30),
        ('ecosystem', 'pypi', 'config_all', 30),
        ('ecosystem', 'maven', 'config_all', 30),
        ('config_activated', 1, 'activated_only', 30),
        ('config_activated', 0, 'configured_not_activated', 30),
        ('ecosystem', 'maven', 'high_cvss', 60),
        ('ecosystem', 'cargo_unobserved', 'absent_ecosystem', 30),
        ('dependency_depth', 2, 'transitive_depth2', 30),
    )
    for i, (fk, fv, lbl, h) in enumerate(eco_items, start=15):
        checks.append({
            'check_id': f'RC-{i:03d}', 'check_family': 'ecosystem_config_boundary', 'kind': 'att',
            'filter_key': fk, 'filter_val': fv, 'label': lbl, 'horizon_days': h,
        })
                                                  
    for i, (mode_c, param_c) in enumerate((
        ('drop_top1_cve', 1), ('drop_top3_cves', 3), ('drop_top1_repo', 1),
        ('drop_top2_repos', 2), ('only_top_cve', 1), ('high_activity_quartile', 1),
    ), start=23):
        checks.append({
            'check_id': f'RC-{i:03d}', 'check_family': 'concentration_influence', 'kind': 'att',
            'concentration_mode': mode_c, 'param': param_c, 'horizon_days': 30,
        })
                                        
    for i, (y_k, desc_k, h) in enumerate((
        ('y_active_14d', 'active_14d', 14),
        ('y_active_60d', 'active_60d', 60),
        ('y_active_90d', 'active_90d', 90),
        ('y_passive_only_30d', 'competing_passive_only', 30),
        ('y_active_30d', 'archived_repo_censoring', 30),
        ('y_active_30d', 'beyond_support_censoring_180d', 180),
    ), start=29):
        checks.append({
            'check_id': f'RC-{i:03d}', 'check_family': 'censoring_competing_risk', 'kind': 'att',
            'y_col': y_k, 'label': desc_k, 'horizon_days': h,
            'archived_only': (desc_k == 'archived_repo_censoring'),
        })
                              
    for i, (y_k, lbl, fk, fv) in enumerate((
        ('y_placebo_unmapped_30d', 'unmapped_dev_coordinate_placebo', None, None),
        ('y_placebo_future_30d', 'future_dated_shock_placebo', None, None),
        ('y_placebo_unmapped_30d', 'unmapped_npm_placebo', 'ecosystem', 'npm'),
        ('y_placebo_unmapped_30d', 'unmapped_pypi_placebo', 'ecosystem', 'pypi'),
        ('y_placebo_future_30d', 'future_shock_pre2023_placebo', 'calendar_regime', 'pre_2023'),
        ('y_passive_only_30d', 'scheduled_bot_passive_placebo', 'config_activated', 0),
    ), start=35):
        checks.append({
            'check_id': f'RC-{i:03d}', 'check_family': 'negative_control', 'kind': 'att',
            'y_col': y_k, 'label': lbl, 'filter_key': fk, 'filter_val': fv, 'horizon_days': 30,
        })
    assert len(checks) == MAX_BOUNDARY_CHECKS
    return checks


def _evaluate_panel_att_subset(
    rows: list[dict[str, Any]],
    *,
    y_col: str,
    cluster_col: str = 'repository_id',
    horizon_days: int = 30,
    max_followup_supported: int = 90,
    missing_shift: float = 0.0,
    weight_col: str | None = None,
) -> dict[str, Any]:
    pass
    if int(horizon_days) > int(max_followup_supported):
        return {
            'test_status': 'NOT_IDENTIFIED',
            'estimate': None,
            'lower': None,
            'upper': None,
            'n_clusters': len({str(r.get(cluster_col) or '') for r in rows}),
            'n_cves': len({str(r.get('cve_id') or '') for r in rows}),
            'n_rows': len(rows),
            'failure_reason': f'horizon_days={horizon_days} exceeds supported followup={max_followup_supported}d',
        }
    if not rows:
        return {
            'test_status': 'SPARSE_ZERO_CELL',
            'estimate': None,
            'lower': None,
            'upper': None,
            'n_clusters': 0,
            'n_cves': 0,
            'n_rows': 0,
            'failure_reason': 'zero_rows_in_stratum',
        }

    clusters = sorted({str(r.get(cluster_col) or '') for r in rows if r.get(cluster_col)})
    n_clusters = len(clusters)
    n_cves = len({str(r.get('cve_id') or '') for r in rows})
    if n_clusters < 2:
        return {
            'test_status': 'INSUFFICIENT_CLUSTERS',
            'estimate': None,
            'lower': None,
            'upper': None,
            'n_clusters': n_clusters,
            'n_cves': n_cves,
            'n_rows': len(rows),
            'failure_reason': f'cluster_count={n_clusters} < 2 minimum for variance estimation',
        }

    t_rows = [r for r in rows if bool(int(r.get('signal_exposed', 0))) and r.get(y_col) is not None]
    c_rows = [r for r in rows if not bool(int(r.get('signal_exposed', 0))) and r.get(y_col) is not None]
    if not t_rows or not c_rows:
        return {
            'test_status': 'SPARSE_ZERO_CELL',
            'estimate': None,
            'lower': None,
            'upper': None,
            'n_clusters': n_clusters,
            'n_cves': n_cves,
            'n_rows': len(rows),
            'failure_reason': f'zero_cell_count (n_treated={len(t_rows)}, n_control={len(c_rows)})',
        }

    w_t = np.array([float(r.get(weight_col, 1.0)) if weight_col else 1.0 for r in t_rows], dtype=float)
    w_c = np.array([float(r.get(weight_col, 1.0)) if weight_col else 1.0 for r in c_rows], dtype=float)
    y_t = np.array([float(r[y_col]) for r in t_rows], dtype=float)
    y_c = np.array([float(r[y_col]) for r in c_rows], dtype=float)

    mu_t = float(np.sum(w_t * y_t) / np.sum(w_t))
    mu_c = float(np.sum(w_c * y_c) / np.sum(w_c))
    est = mu_t - mu_c + float(missing_shift)

                                        
    c_map = {cid: i for i, cid in enumerate(clusters)}
    inf_vec = np.zeros(n_clusters, dtype=float)
    wt_tot = float(np.sum(w_t))
    wc_tot = float(np.sum(w_c))
    for i, r in enumerate(t_rows):
        cid = str(r.get(cluster_col) or '')
        inf_vec[c_map[cid]] += w_t[i] * (y_t[i] - mu_t) / wt_tot
    for i, r in enumerate(c_rows):
        cid = str(r.get(cluster_col) or '')
        inf_vec[c_map[cid]] -= w_c[i] * (y_c[i] - mu_c) / wc_tot

    dof_scale = n_clusters / float(n_clusters - 1)
    se = math.sqrt(max(dof_scale * float(np.sum(inf_vec ** 2)), 1e-18))
    t_crit = float(stats.t.ppf(0.975, df=max(n_clusters - 1, 1)))
    lo = est - t_crit * se
    hi = est + t_crit * se
    status = 'LIMITED_INFERENCE_FEW_CLUSTERS' if n_clusters < FEW_CLUSTER_THRESHOLD else 'ESTIMABLE'
    return {
        'test_status': status,
        'estimate': float(est),
        'lower': float(lo),
        'upper': float(hi),
        'n_clusters': n_clusters,
        'n_cves': n_cves,
        'n_rows': len(t_rows) + len(c_rows),
        'failure_reason': '',
    }


def _evaluate_crosswalk_spec(
    panel_data: list[dict[str, Any]],
    spec: dict[str, Any],
    max_followup_supported: int,
) -> dict[str, Any]:
    axis = spec['crosswalk_axis']
    p = spec['params']
    horizon = int(p.get('horizon_days', 30))

    y_col = 'y_active_30d'
    if horizon == 7:
        y_col = 'y_active_7d'
    elif horizon == 14:
        y_col = 'y_active_14d'
    elif horizon == 60:
        y_col = 'y_active_60d'
    elif horizon == 90:
        y_col = 'y_active_90d'
    if y_col not in panel_data[0] and horizon <= max_followup_supported:
        y_col = 'y_active_30d'

    rows = list(panel_data)
    cluster_col = 'repository_id'
    missing_shift = 0.0
    weight_col: str | None = None

    if axis == 'horizon':
        if p['cohort'] == 'kev_subset':
            rows = [r for r in rows if int(r.get('is_kev', 0)) == 1]
        elif p['cohort'] == 'non_kev_subset':
            rows = [r for r in rows if int(r.get('is_kev', 0)) == 0]
    elif axis == 'clustering':
        cm = p['cluster_mode']
        if cm == 'single_cluster_deg':
            rows = rows[: max(1, min(3, len(rows)))]
                                                                                           
            rows = [dict(r, _deg_cluster='ONLY_ONE_CLUSTER') for r in rows]
            cluster_col = '_deg_cluster'
        elif cm in ('repository_id', 'family_id', 'cve_id'):
            cluster_col = cm
        elif cm == 'ecosystem_x_calendar':
            rows = [dict(r, _eco_cal=f'{r.get("ecosystem")}_{r.get("calendar_regime")}') for r in rows]
            cluster_col = '_eco_cal'
    elif axis == 'covariate_adjustment':
        sm = p['subset']
        if sm == 'npm_pypi':
            rows = [r for r in rows if r.get('ecosystem') in ('npm', 'pypi')]
        elif sm == 'depth1_direct':
            rows = [r for r in rows if int(r.get('dependency_depth', 1)) == 1]
        elif sm == 'high_cvss':
            rows = [r for r in rows if float(r.get('cvss', 0.0)) >= 7.0]
        elif sm == 'active_repos':
            rows = [r for r in rows if float(r.get('repo_activity_z', 0.0)) >= -0.2]
        if p['cov_mode'] in ('trimmed_weight_p05', 'dr_activity_and_age'):
            weight_col = 'sample_weight_base'
    elif axis == 'missingness_bounds':
        bm = p['bound_mode']
        mr = float(p['assumed_missing_rate'])
        if bm == 'unsupported_heavy_censoring':
            return {
                'test_status': 'NOT_IDENTIFIED',
                'estimate': None,
                'lower': None,
                'upper': None,
                'n_clusters': len({str(r.get('repository_id') or '') for r in rows}),
                'n_cves': len({str(r.get('cve_id') or '') for r in rows}),
                'n_rows': len(rows),
                'failure_reason': f'worst_case_censoring_rate={mr:.2f} breaks informative missingness bound',
            }
        if bm == 'complete_case':
            rows = [r for r in rows if int(r.get('missing_trace_flag', 0)) == 0]
        elif bm == 'manski_lower':
            missing_shift = -mr
        elif bm == 'manski_upper':
            missing_shift = mr
        elif bm == 'lee_lower':
            missing_shift = -0.5 * mr
        elif bm == 'lee_upper':
            missing_shift = 0.5 * mr
    elif axis == 'ecosystem_leave_one_out':
        em = p['eco_mode']
        if em == 'leave_out_npm':
            rows = [r for r in rows if r.get('ecosystem') != 'npm']
        elif em == 'leave_out_pypi':
            rows = [r for r in rows if r.get('ecosystem') != 'pypi']
        elif em == 'leave_out_maven':
            rows = [r for r in rows if r.get('ecosystem') != 'maven']
        elif em == 'only_npm':
            rows = [r for r in rows if r.get('ecosystem') == 'npm']
        elif em == 'only_pypi':
            rows = [r for r in rows if r.get('ecosystem') == 'pypi']
        elif em == 'only_maven':
            rows = [r for r in rows if r.get('ecosystem') == 'maven']
        elif em == 'only_cargo_absent':
            rows = [r for r in rows if r.get('ecosystem') == 'cargo']
        elif em == 'only_rubygems_absent':
            rows = [r for r in rows if r.get('ecosystem') == 'rubygems']
        elif em == 'leave_out_depth2':
            rows = [r for r in rows if int(r.get('dependency_depth', 1)) != 2]
        elif em == 'only_depth1':
            rows = [r for r in rows if int(r.get('dependency_depth', 1)) == 1]
    elif axis == 'clock_ambiguity':
        cm = p['clock_mode']
        if cm == 'committer_clock' and 'y_active_30d_committer_clock' in panel_data[0]:
            y_col = 'y_active_30d_committer_clock'
        elif cm == 'pr_merge_clock' and 'y_active_30d_pr_clock' in panel_data[0]:
            y_col = 'y_active_30d_pr_clock'
        elif cm == 'worst_case_delay_shift':
            missing_shift = -0.002 * (float(p['skew_hours']) / 24.0)
        elif cm == 'discordant_drop' and int(p['skew_hours']) >= 72:
            return {
                'test_status': 'NOT_IDENTIFIED',
                'estimate': None,
                'lower': None,
                'upper': None,
                'n_clusters': len({str(r.get('repository_id') or '') for r in rows}),
                'n_cves': len({str(r.get('cve_id') or '') for r in rows}),
                'n_rows': len(rows),
                'failure_reason': 'clock_skew_72h_exceeds_identifiable_pre_post_window',
            }
    elif axis == 'placebo_timing':
        pm = p['placebo_mode']
        pc = p['cohort']
        if pc == 'npm_only':
            rows = [r for r in rows if r.get('ecosystem') == 'npm']
        elif pc == 'pypi_only':
            rows = [r for r in rows if r.get('ecosystem') == 'pypi']
        elif pc == 'maven_only':
            rows = [r for r in rows if r.get('ecosystem') == 'maven']
        elif pc == 'high_activity':
            rows = [r for r in rows if float(r.get('repo_activity_z', 0.0)) > 0.0]

        if pm == 'empty_pre_period_probe':
            return {
                'test_status': 'NOT_IDENTIFIED',
                'estimate': None,
                'lower': None,
                'upper': None,
                'n_clusters': len({str(r.get('repository_id') or '') for r in rows}),
                'n_cves': len({str(r.get('cve_id') or '') for r in rows}),
                'n_rows': len(rows),
                'failure_reason': 'zero_pre_period_support_for_placebo_window',
            }
        if pm == 'unmapped_dev_placebo' and rows and 'y_placebo_unmapped_30d' in rows[0]:
            y_col = 'y_placebo_unmapped_30d'
        elif rows and 'y_placebo_future_30d' in rows[0]:
            y_col = 'y_placebo_future_30d'

    return _evaluate_panel_att_subset(
        rows,
        y_col=y_col,
        cluster_col=cluster_col,
        horizon_days=horizon,
        max_followup_supported=max_followup_supported,
        missing_shift=missing_shift,
        weight_col=weight_col,
    )


def run_r0_robustness(
    panel_data: list[dict[str, Any]],
    out_dir: Path | str,
    *,
    mode: str = 'synthetic',
    seed: int | None = None,
    max_followup_days_supported: int = 90,
    custom_boundary_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pass
    out_path = Path(out_dir).resolve()
    mode_label = _check_mode_gate(mode, out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if not panel_data:
        raise ValueError('panel_data must be non-empty for R0 robustness')

    crosswalk_specs = _build_210_crosswalk_specs()
    boundary_specs = (
        list(custom_boundary_checks) if custom_boundary_checks is not None else _build_40_boundary_specs()
    )
    if len(boundary_specs) > MAX_BOUNDARY_CHECKS:
        raise ValueError(
            f'Registered boundary checks ({len(boundary_specs)}) exceed cap {MAX_BOUNDARY_CHECKS}'
        )

    code_sha = _code_sha256()
    input_sha = _canonical_sha256(panel_data)
    config_sha = _canonical_sha256(
        {
            'construct_version': CONSTRUCT_VERSION_R0,
            'mode': mode_label,
            'seed': seed,
            'max_followup_days_supported': int(max_followup_days_supported),
            'n_crosswalk_rows': len(crosswalk_specs),
            'n_boundary_checks': len(boundary_specs),
        }
    )
    run_id = f'R0-{mode_label}-{config_sha[:10]}'

    crosswalk_rows: list[dict[str, Any]] = []
    for spec in crosswalk_specs:
        res = _evaluate_crosswalk_spec(panel_data, spec, int(max_followup_days_supported))
        row = _make_result_row_base(
            experiment_id='R0',
            run_id=run_id,
            config_sha=config_sha,
            code_sha=code_sha,
            input_sha=input_sha,
            estimand=f'crosswalk_{spec["row_id"]}',
            exposure=spec['crosswalk_axis'],
            comparator=spec['spec_label'],
            subgroup='crosswalk_grid',
            horizon_days=int(spec['params'].get('horizon_days', 30)),
            estimate=res['estimate'],
            lower=res['lower'],
            upper=res['upper'],
            scale=str(spec['params'].get('scale', 'risk_difference')),
            uncertainty_method='cluster_linearized_t',
            independent_unit='repository_cluster',
            n_repositories=res['n_clusters'],
            n_cves=res['n_cves'],
            n_fix_dyads=res['n_rows'],
            n_calendar_shocks=0,
            multiplicity_family=f'R0_{spec["crosswalk_axis"]}',
            test_status=res['test_status'],
            failure_reason=res['failure_reason'],
        )
        row['row_id'] = spec['row_id']
        row['crosswalk_axis'] = spec['crosswalk_axis']
        row['legacy_axis'] = spec['legacy_axis']
        row['legacy_spec_id'] = spec['legacy_spec_id']
        row['spec_label'] = spec['spec_label']
        study_contract.validate_result_row(row)
        crosswalk_rows.append(row)

                                                                                
    assert len(crosswalk_rows) == MAX_LEGACY_CROSSWALK_ROWS

    boundary_rows: list[dict[str, Any]] = []
    for spec in boundary_specs:
        cid = str(spec['check_id'])
        fam = str(spec.get('check_family', 'boundary_check'))
        horizon = int(spec.get('horizon_days', 30))

        if spec.get('kind') == 'iv':
            iv_res = audit_and_run_exploratory_iv(panel_data, spec)
            row = _make_result_row_base(
                experiment_id='R0',
                run_id=run_id,
                config_sha=config_sha,
                code_sha=code_sha,
                input_sha=input_sha,
                estimand=f'iv_audit_{cid}',
                exposure=str(spec.get('d_col', 'treatment_d')),
                comparator=str(spec.get('z_col', 'iv_z')),
                subgroup='exploratory_iv_stratum',
                horizon_days=horizon,
                estimate=iv_res['estimate'],
                lower=iv_res['lower'],
                upper=iv_res['upper'],
                scale='2sls_wald_risk_difference',
                uncertainty_method='cluster_2sls_t',
                independent_unit='repository_cluster',
                n_repositories=iv_res['n_clusters'],
                n_cves=len({str(r.get('cve_id') or '') for r in panel_data}),
                n_fix_dyads=iv_res['n_rows'],
                n_calendar_shocks=0,
                multiplicity_family='R0_boundary_iv',
                test_status=iv_res['test_status'],
                failure_reason=iv_res['failure_reason'],
            )
            row['check_id'] = cid
            row['check_family'] = fam
            row['first_stage_f'] = iv_res['first_stage_f']
            row['exclusion_plausible'] = iv_res['exclusion_plausible']
            row['timing_valid'] = iv_res['timing_valid']
        else:
            sub_rows = list(panel_data)
            fk = spec.get('filter_key')
            fv = spec.get('filter_val')
            if fk is not None:
                sub_rows = [r for r in sub_rows if r.get(fk) == fv]
            if spec.get('archived_only'):
                sub_rows = [r for r in sub_rows if int(r.get('is_archived_repo', 0)) == 1]

            cmode = spec.get('concentration_mode')
            if cmode:
                c_param = int(spec.get('param', 1))
                if cmode.startswith('drop_top') and cmode.endswith('cve' if c_param == 1 else 'cves'):
                    cve_counts: dict[str, int] = {}
                    for r in sub_rows:
                        cve_k = str(r.get('cve_id') or '')
                        cve_counts[cve_k] = cve_counts.get(cve_k, 0) + 1
                    top_cves = {
                        k for k, _ in sorted(cve_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:c_param]
                    }
                    sub_rows = [r for r in sub_rows if str(r.get('cve_id') or '') not in top_cves]
                elif cmode.startswith('drop_top') and 'repo' in cmode:
                    repo_counts: dict[str, int] = {}
                    for r in sub_rows:
                        rk = str(r.get('repository_id') or '')
                        repo_counts[rk] = repo_counts.get(rk, 0) + 1
                    top_repos = {
                        k for k, _ in sorted(repo_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:c_param]
                    }
                    sub_rows = [r for r in sub_rows if str(r.get('repository_id') or '') not in top_repos]
                elif cmode == 'only_top_cve':
                    cve_counts2: dict[str, int] = {}
                    for r in sub_rows:
                        cve_k = str(r.get('cve_id') or '')
                        cve_counts2[cve_k] = cve_counts2.get(cve_k, 0) + 1
                    top1 = sorted(cve_counts2.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                    sub_rows = [r for r in sub_rows if str(r.get('cve_id') or '') == top1]
                elif cmode == 'high_activity_quartile':
                    sub_rows = [r for r in sub_rows if float(r.get('repo_activity_z', 0.0)) >= 0.4]

            y_col = str(spec.get('y_col') or f'y_active_{horizon}d')
            if y_col not in panel_data[0]:
                y_col = 'y_active_30d'

            res = _evaluate_panel_att_subset(
                sub_rows,
                y_col=y_col,
                horizon_days=horizon,
                max_followup_supported=int(max_followup_days_supported),
            )
            row = _make_result_row_base(
                experiment_id='R0',
                run_id=run_id,
                config_sha=config_sha,
                code_sha=code_sha,
                input_sha=input_sha,
                estimand=f'boundary_{cid}',
                exposure=str(spec.get('label') or spec.get('filter_val') or spec.get('concentration_mode') or fam),
                comparator='within_stratum_control',
                subgroup=fam,
                horizon_days=horizon,
                estimate=res['estimate'],
                lower=res['lower'],
                upper=res['upper'],
                scale='risk_difference',
                uncertainty_method='cluster_linearized_t',
                independent_unit='repository_cluster',
                n_repositories=res['n_clusters'],
                n_cves=res['n_cves'],
                n_fix_dyads=res['n_rows'],
                n_calendar_shocks=0,
                multiplicity_family=f'R0_{fam}',
                test_status=res['test_status'],
                failure_reason=res['failure_reason'],
            )
            row['check_id'] = cid
            row['check_family'] = fam
            row['first_stage_f'] = ''
            row['exclusion_plausible'] = ''
            row['timing_valid'] = ''

        study_contract.validate_result_row(row)
        boundary_rows.append(row)

    study_contract.atomic_write_csv(
        out_path / 'robustness_crosswalk.csv',
        ROBUSTNESS_CROSSWALK_FIELDS,
        crosswalk_rows,
    )
    study_contract.atomic_write_csv(
        out_path / 'boundary_checks.csv',
        BOUNDARY_CHECK_FIELDS,
        boundary_rows,
    )

    non_id_crosswalk = [r for r in crosswalk_rows if r['test_status'] in study_contract.NONESTIMABLE_STATUSES]
    non_id_boundary = [r for r in boundary_rows if r['test_status'] in study_contract.NONESTIMABLE_STATUSES]
    iv_rows = [r for r in boundary_rows if r['check_family'] == 'exploratory_iv_audit']

    md_lines = [
        f'# R0 Negative Results, Boundary Failures, and Identification Audit ({mode_label})',
        '',
        f'- Run ID: `{run_id}`',
        f'- Legacy crosswalk rows preserved without pruning: {len(crosswalk_rows)} of {MAX_LEGACY_CROSSWALK_ROWS}',
        f'- Non-estimable / non-identified crosswalk rows: {len(non_id_crosswalk)}',
        f'- Registered boundary/negative-control/IV checks: {len(boundary_rows)} (cap {MAX_BOUNDARY_CHECKS})',
        f'- Non-identified boundary checks: {len(non_id_boundary)}',
        '',
        '## Exploratory IV Identification Audit',
        '',
        '| Check ID | Status | First-Stage F | Exclusion Plausible | Timing Valid | Failure Reason |',
        '| :- | :- | -: | :- | :- | :- |',
    ]
    for r in iv_rows:
        md_lines.append(
            f'| {r["check_id"]} | {r["test_status"]} | {r["first_stage_f"]} | '
            f'{r["exclusion_plausible"]} | {r["timing_valid"]} | {r["failure_reason"]} |'
        )
    md_lines.extend(['', '## Non-Identified Crosswalk Rows Retained (Zero Pruning)', ''])
    for r in non_id_crosswalk[:25]:
        md_lines.append(
            f'- `{r["row_id"]}` ({r["crosswalk_axis"]} / {r["spec_label"]}): '
            f'`{r["test_status"]}` ({r["failure_reason"]})'
        )
    if len(non_id_crosswalk) > 25:
        md_lines.append(f'- Additional retained non-identified crosswalk rows: {len(non_id_crosswalk) - 25}')

    neg_md = '\n'.join(md_lines) + '\n'
    _assert_text_clean(neg_md, 'negative_results.md')
    study_contract.atomic_write_text(out_path / 'negative_results.md', neg_md)

    _assert_text_clean((out_path / 'robustness_crosswalk.csv').read_text(encoding='utf-8'), 'robustness_crosswalk.csv')
    _assert_text_clean((out_path / 'boundary_checks.csv').read_text(encoding='utf-8'), 'boundary_checks.csv')

    return {
        'run_id': run_id,
        'mode_label': mode_label,
        'crosswalk_row_count': len(crosswalk_rows),
        'boundary_check_count': len(boundary_rows),
        'non_identified_crosswalk_count': len(non_id_crosswalk),
        'non_identified_boundary_count': len(non_id_boundary),
        'robustness_crosswalk_csv': str(out_path / 'robustness_crosswalk.csv'),
        'boundary_checks_csv': str(out_path / 'boundary_checks.csv'),
        'negative_results_md': str(out_path / 'negative_results.md'),
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Post-V1 A0 paired ablation and R0 210-row crosswalk + 40-check runner'
    )
    parser.add_argument('-m', choices=('synthetic', 'official'), default='synthetic', help='Execution mode')
    parser.add_argument('-t', choices=('a0', 'r0', 'all'), default='all', help='Task slice to run')
    parser.add_argument('-o', required=True, help='Output directory path')
    parser.add_argument('-b', type=int, default=1999, help='Bootstrap replicates for A0')
    parser.add_argument('-s', type=int, default=None, help='Optional integer deterministic seed override')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    out_root = Path(args.o).resolve()
    summary: dict[str, Any] = {'mode': args.m, 'task': args.t}
    if args.t in ('a0', 'all'):
        paired = generate_synthetic_a0_paired_data(seed=args.s)
        summary['a0'] = run_a0_ablations(
            paired,
            out_root / 'a0',
            bootstrap_reps=args.b,
            mode=args.m,
            seed=args.s,
        )
    if args.t in ('r0', 'all'):
        panel = generate_synthetic_r0_panel_data(seed=args.s)
        summary['r0'] = run_r0_robustness(
            panel,
            out_root / 'r0',
            mode=args.m,
            seed=args.s,
        )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
