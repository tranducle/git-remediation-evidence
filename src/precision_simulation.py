"""Simulate crossed-cluster precision for fixed effect-size thresholds."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import norm, t as student_t

THIS_DIR = Path(__file__).resolve().parent
EXEC_DIR = THIS_DIR.parent
I0_DIR = Path(__file__).resolve().parent

for _p in (str(THIS_DIR), str(I0_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import study_contract              
import estimators              
import validation              


EXPERIMENT_ID = 'P0'
DEFAULT_MME_TARGETS: tuple[float, ...] = (0.05, 0.04, 0.035)
SENSITIVITY_MME_TARGETS: tuple[float, ...] = (0.01, 0.02)
ALL_MME_TARGETS: tuple[float, ...] = (0.05, 0.04, 0.035, 0.02, 0.01)
DEFAULT_ICCS: tuple[float, ...] = (0.01, 0.05, 0.10)
DEFAULT_ICC_WEIGHTS: tuple[float, float, float] = (0.50, 0.30, 0.20)
DEFAULT_REPS = 1000
DEFAULT_ALPHA = 0.05
SMALL_CVE_CLUSTER_LIMIT = 25
SMALL_SHOCK_CLUSTER_LIMIT = 10

PRECISION_COLUMNS: tuple[str, ...] = (
    'study_id',
    'experiment_id',
    'scenario_id',
    'scenario_label',
    'treat_level',
    'p_baseline',
    'n_obs',
    'n_repos',
    'n_cves',
    'n_shocks',
    'min_cluster_count',
    'small_cluster_warning',
    'icc',
    'mme_target',
    'target_tier',
    'reps_requested',
    'valid_reps',
    'nonpositive_variance_count',
    'mean_cluster_se',
    'median_cluster_se',
    'mean_ci95_half_width',
    'mde_80_detection',
    'mde_80_rule_out',
    'prob_rule_out_mme',
    'detection_power',
    'child_seed',
)


DEFAULT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        'scenario_id': 'primary_full_cohort',
        'scenario_label': 'Primary full cohort (repo-level bot assignment)',
        'n_repos': 1200,
        'n_cves': 180,
        'n_shocks': 24,
        'obs_per_repo': 8,
        'p_baseline': 0.50,
        'treat_level': 'repo',
        'target_tier_note': 'conservative_p05',
    },
    {
        'scenario_id': 'within_repo_staggered',
        'scenario_label': 'Within-repository staggered bot cohort',
        'n_repos': 800,
        'n_cves': 160,
        'n_shocks': 24,
        'obs_per_repo': 10,
        'p_baseline': 0.50,
        'treat_level': 'repo_cve',
        'target_tier_note': 'conservative_p05',
    },
    {
        'scenario_id': 'moderate_pre_screen_cohort',
        'scenario_label': 'Moderate pre-screened cohort',
        'n_repos': 400,
        'n_cves': 80,
        'n_shocks': 16,
        'obs_per_repo': 8,
        'p_baseline': 0.50,
        'treat_level': 'repo',
        'target_tier_note': 'conservative_p05',
    },
    {
        'scenario_id': 'kev_subgroup_small_cve',
        'scenario_label': 'KEV subgroup (< 25 CVE clusters)',
        'n_repos': 160,
        'n_cves': 18,
        'n_shocks': 8,
        'obs_per_repo': 4,
        'p_baseline': 0.50,
        'treat_level': 'repo',
        'target_tier_note': 'limited_inference',
    },
    {
        'scenario_id': 'rare_outcome_removal_sens',
        'scenario_label': 'Rare outcome removal sensitivity (p=0.05)',
        'n_repos': 800,
        'n_cves': 160,
        'n_shocks': 24,
        'obs_per_repo': 8,
        'p_baseline': 0.05,
        'treat_level': 'repo',
        'target_tier_note': 'sensitivity_rare',
    },
)


def _dense_index(arr: np.ndarray) -> tuple[np.ndarray, int]:
    pass
    _, inverse = np.unique(arr, return_inverse=True)
    return inverse.astype(np.int64), int(inverse.max()) + 1


def _encode_pair_indices(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    pass
    multiplier = int(b.max()) + 1
    codes = a.astype(np.int64) * multiplier + b.astype(np.int64)
    return _dense_index(codes)


def _encode_triple_indices(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, int]:
    pass
    m_c = int(c.max()) + 1
    m_b = int(b.max()) + 1
    codes = ((a.astype(np.int64) * m_b) + b.astype(np.int64)) * m_c + c.astype(np.int64)
    return _dense_index(codes)


def build_crossed_cluster_design(
    n_repos: int,
    n_cves: int,
    n_shocks: int,
    obs_per_repo: int,
    treat_level: str = 'repo',
    prop_treated: float = 0.5,
) -> dict[str, Any]:
    pass
    if n_repos < 4 or n_cves < 2 or n_shocks < 2 or obs_per_repo < 1:
        raise ValueError('Insufficient cluster or observation dimensions')
    if treat_level not in ('repo', 'repo_cve'):
        raise ValueError(f'Unsupported treat_level: {treat_level}')

    n_obs = n_repos * obs_per_repo
    obs_idx = np.arange(n_obs, dtype=np.int64)
    raw_repo_idx = obs_idx // obs_per_repo
    slot_idx = obs_idx % obs_per_repo

                                                                            
    stride = 7 if math.gcd(7, n_cves) == 1 else 1
    raw_cve_idx = (raw_repo_idx + slot_idx * max(1, (n_cves // obs_per_repo) + stride)) % n_cves
    if np.unique(raw_cve_idx).size < 2:
        raw_cve_idx = obs_idx % n_cves
                                                                                 
                                                                         
    raw_shock_idx = (raw_cve_idx + slot_idx + (raw_repo_idx // max(1, n_repos // n_shocks))) % n_shocks
    if np.unique(raw_shock_idx).size < 2:
        raw_shock_idx = (raw_repo_idx + slot_idx) % n_shocks

    n_treat_repos = max(2, min(n_repos - 2, int(round(n_repos * prop_treated))))
    if treat_level == 'repo' or obs_per_repo < 3:
        treated = (raw_repo_idx < n_treat_repos).astype(np.int64)
    else:
                                                                                       
        treated = np.where(
            raw_repo_idx < n_treat_repos,
            slot_idx < (obs_per_repo - 1),
            slot_idx == 0,
        ).astype(np.int64)

    n_treated = int(treated.sum())
    n_control = int(n_obs - n_treated)
    if n_treated < 2 or n_control < 2:
        raise ValueError('Treatment design requires at least 2 treated and 2 control units')

    repo_idx, eff_n_repos = _dense_index(raw_repo_idx)
    cve_idx, eff_n_cves = _dense_index(raw_cve_idx)
    shock_idx, eff_n_shocks = _dense_index(raw_shock_idx)
    rc_idx, n_rc = _encode_pair_indices(repo_idx, cve_idx)
    rs_idx, n_rs = _encode_pair_indices(repo_idx, shock_idx)
    cs_idx, n_cs = _encode_pair_indices(cve_idx, shock_idx)
    rcs_idx, n_rcs = _encode_triple_indices(repo_idx, cve_idx, shock_idx)

    return {
        'n_obs': n_obs,
        'n_repos': eff_n_repos,
        'n_cves': eff_n_cves,
        'n_shocks': eff_n_shocks,
        'obs_per_repo': obs_per_repo,
        'repo_idx': repo_idx,
        'cve_idx': cve_idx,
        'shock_idx': shock_idx,
        'rc_idx': rc_idx,
        'n_rc': n_rc,
        'rs_idx': rs_idx,
        'n_rs': n_rs,
        'cs_idx': cs_idx,
        'n_cs': n_cs,
        'rcs_idx': rcs_idx,
        'n_rcs': n_rcs,
        'treated': treated,
        'n_treated': n_treated,
        'n_control': n_control,
    }


def _single_component_variance(
    influence: np.ndarray,
    labels: np.ndarray,
    adjust: bool = True,
) -> float:
    pass
    _, inverse = np.unique(labels, return_inverse=True)
    groups = int(inverse.max()) + 1
    if groups < 2:
        raise ValueError('At least two independent clusters required')
    totals = np.bincount(inverse, weights=influence)
    scale = (groups / (groups - 1)) if adjust else 1.0
    return float(np.dot(totals, totals) * scale)


def threeway_cluster_variance(
    influence: np.ndarray,
    repo_idx: np.ndarray,
    cve_idx: np.ndarray,
    shock_idx: np.ndarray,
    adjust: bool = True,
) -> float:
    pass
    v_rc_two_way = estimators.cluster_variance(influence, repo_idx, cve_idx, adjust=adjust)
    rs_idx, _ = _encode_pair_indices(repo_idx, shock_idx)
    cs_idx, _ = _encode_pair_indices(cve_idx, shock_idx)
    rcs_idx, _ = _encode_triple_indices(repo_idx, cve_idx, shock_idx)
    v_s = _single_component_variance(influence, shock_idx, adjust=adjust)
    v_rs = _single_component_variance(influence, rs_idx, adjust=adjust)
    v_cs = _single_component_variance(influence, cs_idx, adjust=adjust)
    v_rcs = _single_component_variance(influence, rcs_idx, adjust=adjust)
    return float(v_rc_two_way + v_s - v_rs - v_cs + v_rcs)


def _batch_component_variance(
    influence_mat: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    adjust: bool = True,
) -> np.ndarray:
    pass
    dense_idx, eff_groups = _dense_index(group_idx)
    if eff_groups < 2:
        raise ValueError('At least two independent clusters required')
    reps = influence_mat.shape[0]
    totals = np.zeros((eff_groups, reps), dtype=float)
    np.add.at(totals, dense_idx, influence_mat.T)
    scale = (eff_groups / (eff_groups - 1)) if adjust else 1.0
    return np.sum(totals * totals, axis=0) * scale


def _batch_threeway_cgm_variance(
    influence_mat: np.ndarray,
    design: dict[str, Any],
    adjust: bool = True,
) -> np.ndarray:
    pass
    v_r = _batch_component_variance(influence_mat, design['repo_idx'], design['n_repos'], adjust=adjust)
    v_c = _batch_component_variance(influence_mat, design['cve_idx'], design['n_cves'], adjust=adjust)
    v_s = _batch_component_variance(influence_mat, design['shock_idx'], design['n_shocks'], adjust=adjust)
    v_rc = _batch_component_variance(influence_mat, design['rc_idx'], design['n_rc'], adjust=adjust)
    v_rs = _batch_component_variance(influence_mat, design['rs_idx'], design['n_rs'], adjust=adjust)
    v_cs = _batch_component_variance(influence_mat, design['cs_idx'], design['n_cs'], adjust=adjust)
    v_rcs = _batch_component_variance(influence_mat, design['rcs_idx'], design['n_rcs'], adjust=adjust)
    return v_r + v_c + v_s - v_rc - v_rs - v_cs + v_rcs


def _influence_batch(y_mat: np.ndarray, design: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pass
    t_mask = design['treated'] == 1
    c_mask = ~t_mask
    n1 = float(design['n_treated'])
    n0 = float(design['n_control'])
    mean1 = y_mat[:, t_mask].mean(axis=1, keepdims=True)
    mean0 = y_mat[:, c_mask].mean(axis=1, keepdims=True)
    tau = (mean1 - mean0).ravel()

    influence = np.empty_like(y_mat, dtype=float)
    influence[:, t_mask] = (y_mat[:, t_mask] - mean1) / n1
    influence[:, c_mask] = -(y_mat[:, c_mask] - mean0) / n0
    return tau, influence


def _classify_small_cluster_warning(n_cves: int, n_shocks: int, n_repos: int) -> str:
    pass
    reasons: list[str] = []
    if n_cves < SMALL_CVE_CLUSTER_LIMIT:
        reasons.append('LIMITED_SMALL_CVE_CLUSTERS')
    if n_shocks < SMALL_SHOCK_CLUSTER_LIMIT:
        reasons.append('LIMITED_SMALL_SHOCK_CLUSTERS')
    if n_repos < SMALL_CVE_CLUSTER_LIMIT:
        reasons.append('LIMITED_SMALL_REPO_CLUSTERS')
    return '|'.join(reasons) if reasons else 'ADEQUATE_CLUSTER_SUPPORT'


def simulate_scenario_icc(
    scenario: dict[str, Any],
    icc: float,
    mme_targets: Sequence[float] = ALL_MME_TARGETS,
    reps: int = DEFAULT_REPS,
    alpha: float = DEFAULT_ALPHA,
    icc_weights: tuple[float, float, float] = DEFAULT_ICC_WEIGHTS,
    master_seed: int = study_contract.MASTER_SEED,
    adjust: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pass
    if reps < 1:
        raise ValueError(f'reps must be >= 1, got {reps}')
    if not (0.0 <= icc < 1.0):
        raise ValueError(f'icc must lie in [0, 1), got {icc}')
    if abs(sum(icc_weights) - 1.0) > 1e-9:
        raise ValueError('icc_weights must sum to 1.0')

    design = build_crossed_cluster_design(
        n_repos=int(scenario['n_repos']),
        n_cves=int(scenario['n_cves']),
        n_shocks=int(scenario['n_shocks']),
        obs_per_repo=int(scenario['obs_per_repo']),
        treat_level=str(scenario.get('treat_level', 'repo')),
    )

    study_key = (
        study_contract.STUDY_ID
        if master_seed == study_contract.MASTER_SEED
        else f'{study_contract.STUDY_ID}_{master_seed}'
    )
                                                                            
                                                                                       
    crn_seed = study_contract.derive_child_seed(
        study_key,
        EXPERIMENT_ID,
        str(scenario['scenario_id']),
        replicate=0,
    )
    child_seed = study_contract.derive_child_seed(
        study_key,
        EXPERIMENT_ID,
        f"{scenario['scenario_id']}_icc_{icc:.4f}",
        replicate=0,
    )
    rng = np.random.default_rng(crn_seed)

    w_r, w_c, w_s = icc_weights
    sd_r = math.sqrt(icc * w_r)
    sd_c = math.sqrt(icc * w_c)
    sd_s = math.sqrt(icc * w_s)
    sd_e = math.sqrt(max(0.0, 1.0 - icc))

    u_r = rng.standard_normal((reps, design['n_repos']))[:, design['repo_idx']]
    v_c = rng.standard_normal((reps, design['n_cves']))[:, design['cve_idx']]
    w_s_eff = rng.standard_normal((reps, design['n_shocks']))[:, design['shock_idx']]
    eps = rng.standard_normal((reps, design['n_obs']))

    latent = sd_r * u_r + sd_c * v_c + sd_s * w_s_eff + sd_e * eps

    p_base = float(scenario.get('p_baseline', 0.50))
    thresh_0 = float(norm.ppf(1.0 - p_base))
    active_dims = [int(design['n_repos']), int(design['n_cves'])]
    if int(design.get('n_shocks', 0)) >= 2:
        active_dims.append(int(design['n_shocks']))
    df_crit = max(1, min(active_dims) - 1)
    t_alpha = float(student_t.ppf(1.0 - alpha / 2.0, df=df_crit))
    t_beta = float(student_t.ppf(0.80, df=df_crit))

    y_null = (latent > thresh_0).astype(float)
    tau_null, inf_null = _influence_batch(y_null, design)
    var_null = _batch_threeway_cgm_variance(inf_null, design, adjust=adjust)

                                                                               
    if reps >= 1:
        _, inf_t = estimators.weighted_mean_influence(
            y_null[0],
            design['treated'] == 1,
            np.ones(design['n_obs'], dtype=float),
        )
        _, inf_c = estimators.weighted_mean_influence(
            y_null[0],
            design['treated'] == 0,
            np.ones(design['n_obs'], dtype=float),
        )
        v_check = threeway_cluster_variance(
            inf_t - inf_c,
            design['repo_idx'],
            design['cve_idx'],
            design['shock_idx'],
            adjust=adjust,
        )
        if not math.isclose(float(var_null[0]), v_check, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(
                f'Batch variance {var_null[0]} diverges from estimators anchor {v_check}'
            )

    valid_mask = np.isfinite(var_null) & (var_null > 0.0)
    nonpositive_count = int(np.sum(~valid_mask))
    valid_reps = int(np.sum(valid_mask))
    if valid_reps == 0:
        raise RuntimeError(f"All {reps} replicates produced non-positive CGM3 variance")

    se_valid = np.sqrt(var_null[valid_mask])
    tau_valid = tau_null[valid_mask]
    mean_se = float(np.mean(se_valid))
    median_se = float(np.median(se_valid))
    mean_half_width = t_alpha * mean_se
    mde_80_detection = (t_alpha + t_beta) * mean_se
    rule_out_bound = np.abs(tau_valid) + t_alpha * se_valid
    mde_80_rule_out = float(np.quantile(rule_out_bound, 0.80))

    warning_flag = _classify_small_cluster_warning(
        design['n_cves'],
        design['n_shocks'],
        design['n_repos'],
    )
    min_cluster_count = min(design['n_repos'], design['n_cves'], design['n_shocks'])
    t_mask = design['treated'] == 1

    rows: list[dict[str, Any]] = []
    for mme in mme_targets:
        mme_val = float(mme)
        tier = (
            'primary_mme'
            if any(math.isclose(mme_val, target, abs_tol=1e-9) for target in DEFAULT_MME_TARGETS)
            else 'sensitivity_mme'
        )
        prob_rule_out = float(np.mean(rule_out_bound < mme_val))

        p_alt = min(0.9999, p_base + mme_val)
        thresh_1 = float(norm.ppf(1.0 - p_alt))
        thresh_vec = np.where(t_mask, thresh_1, thresh_0)[None, :]
        y_alt = (latent > thresh_vec).astype(float)
        tau_alt, inf_alt = _influence_batch(y_alt, design)
        var_alt = _batch_threeway_cgm_variance(inf_alt, design, adjust=adjust)
        alt_valid = np.isfinite(var_alt) & (var_alt > 0.0)
        if np.any(alt_valid):
            z_stats = np.abs(tau_alt[alt_valid]) / np.sqrt(var_alt[alt_valid])
            detection_power = float(np.mean(z_stats > t_alpha))
        else:
            detection_power = 0.0

        rows.append(
            {
                'study_id': study_contract.STUDY_ID,
                'experiment_id': EXPERIMENT_ID,
                'scenario_id': scenario['scenario_id'],
                'scenario_label': scenario['scenario_label'],
                'treat_level': scenario.get('treat_level', 'repo'),
                'p_baseline': f'{p_base:.4f}',
                'n_obs': design['n_obs'],
                'n_repos': design['n_repos'],
                'n_cves': design['n_cves'],
                'n_shocks': design['n_shocks'],
                'min_cluster_count': min_cluster_count,
                'small_cluster_warning': warning_flag,
                'icc': f'{icc:.4f}',
                'mme_target': f'{mme_val:.4f}',
                'target_tier': tier,
                'reps_requested': reps,
                'valid_reps': valid_reps,
                'nonpositive_variance_count': nonpositive_count,
                'mean_cluster_se': f'{mean_se:.6f}',
                'median_cluster_se': f'{median_se:.6f}',
                'mean_ci95_half_width': f'{mean_half_width:.6f}',
                'mde_80_detection': f'{mde_80_detection:.6f}',
                'mde_80_rule_out': f'{mde_80_rule_out:.6f}',
                'prob_rule_out_mme': f'{prob_rule_out:.4f}',
                'detection_power': f'{detection_power:.4f}',
                'child_seed': child_seed,
            }
        )

    seed_record = {
        'scenario_id': scenario['scenario_id'],
        'icc': icc,
        'child_seed': child_seed,
        'reps_requested': reps,
        'valid_reps': valid_reps,
        'nonpositive_variance_count': nonpositive_count,
        'small_cluster_warning': warning_flag,
    }
    return rows, seed_record


def build_scenarios_from_cluster_summary(
    cluster_summary: dict[str, Any] | Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    pass
    items = [cluster_summary] if isinstance(cluster_summary, dict) else list(cluster_summary)
    built: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        built.append(
            {
                'scenario_id': str(item.get('scenario_id', f'custom_scenario_{idx}')),
                'scenario_label': str(item.get('scenario_label', item.get('scenario_id', f'Custom scenario {idx}'))),
                'n_repos': int(item['n_repos']),
                'n_cves': int(item['n_cves']),
                'n_shocks': int(item['n_shocks']),
                'obs_per_repo': int(item.get('obs_per_repo', 4)),
                'p_baseline': float(item.get('p_baseline', 0.50)),
                'treat_level': str(item.get('treat_level', 'repo')),
                'target_tier_note': str(item.get('target_tier_note', 'cluster_summary')),
            }
        )
    return built


def _render_precision_report(
    rows: list[dict[str, Any]],
    mode: str,
    reps: int,
    alpha: float,
) -> str:
    pass
    lines: list[str] = [
        '# P0 Prospective Crossed-Cluster Precision and MDE Report',
        '',
        f'- Study ID: `{study_contract.STUDY_ID}`',
        f'- Mode: `{mode}`',
        f'- Replicates per cell: `{reps}`',
        f'- Two-sided alpha: `{alpha}`',
        f'- Primary MME targets: `{DEFAULT_MME_TARGETS}`',
        f'- Sensitivity MME targets: `{SENSITIVITY_MME_TARGETS}`',
        f'- ICC grid: `{DEFAULT_ICCS}`',
        '',
        '## Scenario Summary Table',
        '',
        '| Scenario | p_base | Repos | CVEs | Shocks | ICC | MME | Rule-Out Prob | Power | MDE80 (Detect) | Cluster SE | Small-Cluster Flag |',
        '| - | - | - | - | - | - | - | - | - | - | - | - |',
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario_id']} | {row['p_baseline']} | {row['n_repos']} | "
            f"{row['n_cves']} | {row['n_shocks']} | {row['icc']} | {row['mme_target']} | "
            f"{row['prob_rule_out_mme']} | {row['detection_power']} | "
            f"{row['mde_80_detection']} | {row['mean_cluster_se']} | "
            f"{row['small_cluster_warning']} |"
        )
    lines.extend(
        [
            '',
            '## Methodological Notes',
            '',
            '1. Variance uses three-way CGM clustering over `(repository, cve, calendar_shock)` with `G / (G - 1)` finite-sample scaling.',
            '2. Replicates with non-positive multiway variance estimates are never silently dropped or clamped to zero; they are counted in `nonpositive_variance_count`.',
            '3. Any scenario with fewer than 25 CVE clusters or fewer than 10 shock clusters is explicitly marked with a limited-inference warning.',
            '',
        ]
    )
    report_text = '\n'.join(lines)
    violations = study_contract.check_no_forbidden_dashes(report_text)
    if violations:
        raise ValueError(f'Forbidden dash detected in generated report: {violations}')
    return report_text


def run_p0_precision(
    out_dir: Path | str,
    reps: int = DEFAULT_REPS,
    mode: str = 'synthetic',
    scenarios: Sequence[dict[str, Any]] | None = None,
    cluster_summary: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    iccs: Sequence[float] = DEFAULT_ICCS,
    mme_targets: Sequence[float] = ALL_MME_TARGETS,
    alpha: float = DEFAULT_ALPHA,
    master_seed: int = study_contract.MASTER_SEED,
    input_paths: Sequence[Path | str] = (),
) -> dict[str, Any]:
    pass
    if mode not in ('synthetic', 'official'):
        raise ValueError(f'Invalid mode: {mode}')

    if mode == 'official':
        validation.assert_official_empirical_unlocked()
    else:
        validation.assert_synthetic_mode_isolated(input_paths)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if scenarios is not None:
        active_scenarios = list(scenarios)
    elif cluster_summary is not None:
        active_scenarios = build_scenarios_from_cluster_summary(cluster_summary)
    else:
        active_scenarios = list(DEFAULT_SCENARIOS)

    all_rows: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    for scenario in active_scenarios:
        for icc in iccs:
            cell_rows, seed_rec = simulate_scenario_icc(
                scenario=scenario,
                icc=float(icc),
                mme_targets=mme_targets,
                reps=reps,
                alpha=alpha,
                master_seed=master_seed,
            )
            all_rows.extend(cell_rows)
            seed_records.append(seed_rec)

    csv_path = out_path / 'precision_scenarios.csv'
    report_path = out_path / 'precision_report.md'
    seeds_path = out_path / 'precision_seeds.json'

    study_contract.atomic_write_csv(csv_path, PRECISION_COLUMNS, all_rows)
    report_text = _render_precision_report(all_rows, mode=mode, reps=reps, alpha=alpha)
    study_contract.atomic_write_text(report_path, report_text)

    seeds_payload = {
        'study_id': study_contract.STUDY_ID,
        'experiment_id': EXPERIMENT_ID,
        'mode': mode,
        'master_seed': master_seed,
        'reps': reps,
        'alpha': alpha,
        'iccs': list(iccs),
        'primary_mme_targets': list(DEFAULT_MME_TARGETS),
        'sensitivity_mme_targets': list(SENSITIVITY_MME_TARGETS),
        'generated_utc': study_contract.utc_now_iso(),
        'cells': seed_records,
    }
    study_contract.atomic_write_json(seeds_path, seeds_payload)

    return {
        'mode': mode,
        'reps': reps,
        'row_count': len(all_rows),
        'scenario_count': len(active_scenarios),
        'csv_path': str(csv_path),
        'report_path': str(report_path),
        'seeds_path': str(seeds_path),
        'rows': all_rows,
        'seeds': seeds_payload,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run P0 cluster-aware prospective precision simulator.',
        add_help=True,
    )
    parser.add_argument(
        '-m',
        dest='mode',
        choices=('synthetic', 'official'),
        default='synthetic',
        help='Execution mode; use scripts/reproduce.py for the released empirical data.',
    )
    parser.add_argument(
        '-o',
        dest='out_dir',
        required=True,
        help='Output directory for precision_scenarios.csv, precision_report.md, and precision_seeds.json.',
    )
    parser.add_argument(
        '-b',
        dest='reps',
        type=int,
        default=DEFAULT_REPS,
        help='Monte Carlo replicate count per (scenario, icc) cell.',
    )
    parser.add_argument(
        '-s',
        dest='seed',
        type=int,
        default=study_contract.MASTER_SEED,
        help='Master seed integer.',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = run_p0_precision(
        out_dir=args.out_dir,
        reps=args.reps,
        mode=args.mode,
        master_seed=args.seed,
    )
    print(
        f"P0 complete: mode={summary['mode']} scenarios={summary['scenario_count']} "
        f"rows={summary['row_count']} csv={summary['csv_path']}"
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
