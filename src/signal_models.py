"""Estimate event-time repair contrasts and competing cumulative incidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.stats import norm, t as student_t

try:
    from . import study_contract, validation
except ImportError:
    import study_contract
    import validation


HERE = Path(__file__).resolve().parent
EXECUTION_ROOT = Path(__file__).resolve().parent.parent
I0_SYNTHETIC_DIR = Path(__file__).resolve().parent


def _load_i0_module(module_name: str) -> Any:
    pass
    target_path = (I0_SYNTHETIC_DIR / f'{module_name}.py').resolve()
    if not target_path.is_file():
        raise ImportError(f'Missing estimator support module: {target_path}')
    alias = f'_i0_{module_name}'
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, str(target_path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to construct import spec for {target_path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_i0_estimators = _load_i0_module('estimators')
_i0_known_dgp = _load_i0_module('synthetic_cohorts')

aj_cif = _i0_estimators.aj_cif
cluster_variance = _i0_estimators.cluster_variance
did_aggregate = _i0_estimators.did_aggregate
did_cell = _i0_estimators.did_cell
weighted_mean_influence = _i0_estimators.weighted_mean_influence
synthetic_cohorts = _i0_known_dgp

MME_DEFAULTS = {
    'E1': 0.05,
    'E2': 0.04,
    'E3': 0.04,
    'E4': 0.035,
}
SENSITIVITY_MME_MARGINS = (0.01, 0.02)
E3_MIN_EXPOSED_CVE_CLUSTERS = 25
E1_MIN_CALENDAR_SHOCKS = 10
DEFAULT_BOOTSTRAP_REPS = 1999
DEFAULT_PERM_REPS = 9999


def _code_sha256() -> str:
    return study_contract.sha256_file(Path(__file__).resolve())


def _hash_json_obj(payload: Any) -> str:
    def _default(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if not math.isfinite(val) else val
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    encoded = json.dumps(payload, sort_keys=True, default=_default, allow_nan=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _resolve_provenance(
    experiment_id: str,
    config_obj: dict[str, Any],
    input_obj: Any,
    config_sha256: str = '',
    input_sha256: str = '',
    run_id: str | None = None,
    mode: str = 'synthetic',
) -> dict[str, str]:
    cfg_sha = config_sha256 or _hash_json_obj(config_obj)
    inp_sha = input_sha256 or _hash_json_obj(input_obj)
    code_sha = _code_sha256()
    resolved_run_id = run_id or f'{mode}_{experiment_id}_{cfg_sha[:12]}'
    return {
        'run_id': resolved_run_id,
        'config_sha256': cfg_sha,
        'code_sha256': code_sha,
        'input_sha256': inp_sha,
    }


def _make_result_row(
    *,
    experiment_id: str,
    provenance: dict[str, str],
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
    failure_reason: str = '',
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        'experiment_id': experiment_id,
        'run_id': provenance['run_id'],
        'config_sha256': provenance['config_sha256'],
        'code_sha256': provenance['code_sha256'],
        'input_sha256': provenance['input_sha256'],
        'estimand': estimand,
        'exposure': exposure,
        'comparator': comparator,
        'subgroup': subgroup,
        'horizon_days': int(horizon_days),
        'estimate': None if estimate is None else float(estimate),
        'lower': None if lower is None else float(lower),
        'upper': None if upper is None else float(upper),
        'scale': scale,
        'uncertainty_method': uncertainty_method,
        'independent_unit': independent_unit,
        'n_repositories': int(n_repositories),
        'n_cves': int(n_cves),
        'n_fix_dyads': int(n_fix_dyads),
        'n_calendar_shocks': int(n_calendar_shocks),
        'multiplicity_family': multiplicity_family,
        'test_status': test_status,
        'failure_reason': str(failure_reason or ''),
    }
    if extra:
        row.update(extra)
    study_contract.validate_result_row(row)
    return row


def _nonestimable_row(
    *,
    experiment_id: str,
    provenance: dict[str, str],
    estimand: str,
    exposure: str,
    comparator: str,
    subgroup: str,
    horizon_days: int,
    scale: str,
    uncertainty_method: str,
    independent_unit: str,
    n_repositories: int,
    n_cves: int,
    n_fix_dyads: int,
    n_calendar_shocks: int,
    multiplicity_family: str,
    status: str,
    failure_reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _make_result_row(
        experiment_id=experiment_id,
        provenance=provenance,
        estimand=estimand,
        exposure=exposure,
        comparator=comparator,
        subgroup=subgroup,
        horizon_days=horizon_days,
        estimate=None,
        lower=None,
        upper=None,
        scale=scale,
        uncertainty_method=uncertainty_method,
        independent_unit=independent_unit,
        n_repositories=n_repositories,
        n_cves=n_cves,
        n_fix_dyads=n_fix_dyads,
        n_calendar_shocks=n_calendar_shocks,
        multiplicity_family=multiplicity_family,
        test_status=status,
        failure_reason=failure_reason,
        extra=extra,
    )


def _unique_count(arr: np.ndarray, mask: np.ndarray | None = None) -> int:
    sub = arr if mask is None else arr[mask]
    if len(sub) == 0:
        return 0
    return int(len(np.unique(sub)))


def _cluster_counts(
    repository: np.ndarray,
    cve: np.ndarray,
    mask: np.ndarray | None = None,
    fix_dyad: np.ndarray | None = None,
    calendar_shock: np.ndarray | None = None,
) -> dict[str, int]:
    sub_mask = np.ones(len(repository), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    n_repos = _unique_count(repository, sub_mask)
    n_cves = _unique_count(cve, sub_mask)
    if fix_dyad is not None:
        n_dyads = _unique_count(fix_dyad, sub_mask)
    else:
        n_dyads = int(np.sum(sub_mask))
    if calendar_shock is not None:
        vals = calendar_shock[sub_mask]
        finite_vals = vals[np.isfinite(vals)] if vals.dtype.kind in 'fcui' else np.array([v for v in vals if v is not None and v != ''])
        n_shocks = int(len(np.unique(finite_vals)))
    else:
        n_shocks = 0
    return {
        'n_repositories': n_repos,
        'n_cves': n_cves,
        'n_fix_dyads': n_dyads,
        'n_calendar_shocks': n_shocks,
    }


def _materiality_diagnostics(
    estimate: float | None,
    lower: float | None,
    upper: float | None,
    mme: float,
    test_status: str,
) -> dict[str, Any]:
    if test_status not in study_contract.ESTIMABLE_STATUSES or estimate is None or lower is None or upper is None:
        return {
            'mme': float(mme),
            'within_mme': None,
            'within_mme_1pp': None,
            'within_mme_2pp': None,
            'exceeds_mme': None,
        }
    return {
        'mme': float(mme),
        'within_mme': bool(abs(lower) < mme and abs(upper) < mme),
        'within_mme_1pp': bool(abs(lower) < 0.01 and abs(upper) < 0.01),
        'within_mme_2pp': bool(abs(lower) < 0.02 and abs(upper) < 0.02),
        'exceeds_mme': bool(lower > mme or upper < -mme),
    }


def _cluster_interval_from_influence(
    estimate: float,
    influence: np.ndarray,
    repository: np.ndarray,
    cve: np.ndarray,
    *,
    bootstrap_method: str = 'two_way_influence',
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = study_contract.MASTER_SEED,
    df_cves_override: int | None = None,
) -> dict[str, Any]:
    pass
    phi = np.asarray(influence, dtype=float)
    repo = np.asarray(repository)
    cve_arr = np.asarray(cve)
    n_repos = _unique_count(repo)
    n_cves = _unique_count(cve_arr)
    if n_repos < 2 or n_cves < 2:
        return {
            'valid': False,
            'status': 'INSUFFICIENT_CLUSTERS',
            'failure_reason': f'At least two repository ({n_repos}) and CVE ({n_cves}) clusters required',
            'se': None,
            'lower': None,
            'upper': None,
            'p_value': None,
            'variance': None,
        }
    try:
        variance = cluster_variance(phi, repo, cve_arr, adjust=True)
    except ValueError as exc:
        return {
            'valid': False,
            'status': 'INSUFFICIENT_CLUSTERS',
            'failure_reason': str(exc),
            'se': None,
            'lower': None,
            'upper': None,
            'p_value': None,
            'variance': None,
        }
    if not math.isfinite(variance) or variance <= 0.0:
        return {
            'valid': False,
            'status': 'NOT_ESTIMABLE',
            'failure_reason': f'Nonpositive two-way cluster variance ({variance}); never silently repaired',
            'se': None,
            'lower': None,
            'upper': None,
            'p_value': None,
            'variance': float(variance) if math.isfinite(variance) else None,
        }
    df_cves = n_cves if df_cves_override is None else max(2, int(df_cves_override))
    df = max(1, min(n_repos, df_cves) - 1)
    crit = float(student_t.ppf(0.975, df))
    se = float(math.sqrt(variance))
    lower = float(estimate - crit * se)
    upper = float(estimate + crit * se)
    p_value = float(2.0 * student_t.sf(abs(estimate / se), df))

    boot_lower = None
    boot_upper = None
    boot_p_value = None
    if bootstrap_method in ('wild_cluster', 'pair_cluster') and bootstrap_reps >= 1:
        _, inv = np.unique(cve_arr, return_inverse=True)
        n_clust = int(inv.max() + 1)
        cluster_sums = np.bincount(inv, weights=phi)
        rng = np.random.default_rng(int(seed))
        if bootstrap_method == 'wild_cluster':
            weights = rng.choice(np.array([-1.0, 1.0]), size=(int(bootstrap_reps), n_clust))
            draws = weights @ cluster_sums
            half = float(np.quantile(np.abs(draws), 0.95))
            boot_lower = float(estimate - half)
            boot_upper = float(estimate + half)
            boot_p_value = float((np.sum(np.abs(draws) >= abs(estimate)) + 1) / (int(bootstrap_reps) + 1))
        else:
            idx = study_contract.paired_cluster_resampling_indices(n_clust, int(bootstrap_reps), int(seed))
            draws = cluster_sums[idx].sum(axis=1)
            centered = draws - float(draws.mean())
            half = float(np.quantile(np.abs(centered), 0.95))
            boot_lower = float(estimate - half)
            boot_upper = float(estimate + half)
            boot_p_value = float((np.sum(np.abs(centered) >= abs(estimate)) + 1) / (int(bootstrap_reps) + 1))
        lower = boot_lower
        upper = boot_upper
        p_value = boot_p_value

    return {
        'valid': True,
        'status': 'ESTIMABLE',
        'failure_reason': '',
        'se': se,
        'lower': lower,
        'upper': upper,
        'p_value': p_value,
        'variance': float(variance),
        'df': df,
        'boot_lower': boot_lower,
        'boot_upper': boot_upper,
        'boot_p_value': boot_p_value,
    }


def _did_event_time_aggregate(
    y: np.ndarray,
    cohort: np.ndarray,
    event_time: int,
    *,
    anticipation: int = 0,
    eligibility: np.ndarray | None = None,
    observed: np.ndarray | None = None,
    observation_probability: np.ndarray | None = None,
    post_selection: bool = False,
) -> dict[str, Any]:
    pass
    n, nt = y.shape
    groups = sorted(int(x) for x in np.unique(cohort) if np.isfinite(x))
    cells: list[dict[str, Any]] = []
    skipped: list[str] = []
    for g in groups:
        t = g + int(event_time)
        baseline = g - int(anticipation) - 1
        if min(t, baseline) < 0 or max(t, baseline) >= nt:
            skipped.append(f'group={g} out_of_window(t={t},base={baseline})')
            continue
        try:
            cell = did_cell(
                y,
                cohort,
                g,
                int(event_time),
                anticipation=anticipation,
                eligibility=eligibility,
                observed=observed,
                observation_probability=observation_probability,
                post_selection=post_selection,
            )
            cells.append(cell)
        except ValueError as exc:
            skipped.append(f'group={g}:{exc}')
    if not cells:
        raise ValueError('No estimable cohort-time DID cells at event_time=' + str(event_time) + (f' ({"; ".join(skipped)})' if skipped else ''))
    counts = np.array([c['treated_count'] for c in cells], dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError('Zero treated support across DID cells')
    weights = counts / total
    estimate = float(sum(w * c['estimate'] for w, c in zip(weights, cells)))
    influence = sum(w * c['influence'] for w, c in zip(weights, cells))
    used = np.any([c['used'] for c in cells], axis=0)
    return {
        'estimate': estimate,
        'influence': influence,
        'weights': weights,
        'cells': cells,
        'used': used,
        'skipped': skipped,
    }


def _cluster_covariance_matrix(
    influences: list[np.ndarray],
    repository: np.ndarray,
    cve: np.ndarray,
) -> np.ndarray:
    pass
    k = len(influences)
    cov = np.zeros((k, k), dtype=float)
    variances = [cluster_variance(influences[i], repository, cve, adjust=True) for i in range(k)]
    for i in range(k):
        cov[i, i] = variances[i]
        for j in range(i + 1, k):
            v_sum = cluster_variance(influences[i] + influences[j], repository, cve, adjust=True)
            c_ij = 0.5 * (v_sum - variances[i] - variances[j])
            cov[i, j] = c_ij
            cov[j, i] = c_ij
    return cov


def honest_did_bounds(
    event_time_rows: list[dict[str, Any]],
    repository: np.ndarray,
    cve: np.ndarray,
    *,
    target_event_time: int = 13,
    reference_event_time: int = -1,
    m_values: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.05),
    mbar_values: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
) -> dict[str, Any]:
    pass
    post_candidates = [r for r in event_time_rows if int(r['event_time']) == int(target_event_time)]
    if not post_candidates or post_candidates[0]['test_status'] not in study_contract.ESTIMABLE_STATUSES:
        return {
            'estimable': False,
            'failure_reason': f'Target event_time={target_event_time} is not estimable for HonestDiD',
            'linear_bounds': [],
            'relative_magnitude_bounds': [],
        }
    post_row = post_candidates[0]
    base_est = float(post_row['estimate'])
    base_se = float(post_row['se'])
    df = int(post_row.get('df') or max(1, min(_unique_count(repository), _unique_count(cve)) - 1))
    crit = float(student_t.ppf(0.975, df))

    pre_rows = sorted(
        [
            r for r in event_time_rows
            if int(r['event_time']) < int(reference_event_time) and r['test_status'] in study_contract.ESTIMABLE_STATUSES
        ],
        key=lambda r: int(r['event_time']),
    )

                                                                                  
    pre_chain = [(int(r['event_time']), float(r['estimate'])) for r in pre_rows] + [(int(reference_event_time), 0.0)]
    pre_chain.sort(key=lambda item: item[0])
    step_diffs = [abs(pre_chain[idx][1] - pre_chain[idx - 1][1]) for idx in range(1, len(pre_chain))]
    max_pre_step = max(step_diffs) if step_diffs else 0.0
    max_pre_abs = max((abs(float(r['estimate'])) for r in pre_rows), default=0.0)
    horizon_steps = max(1, int(target_event_time) - int(reference_event_time))

    rm_bounds = []
    for mbar in mbar_values:
        bias_bound = float(mbar) * horizon_steps * max_pre_step
        half = bias_bound + crit * base_se
        rm_bounds.append({
            'mbar': float(mbar),
            'max_pre_step': float(max_pre_step),
            'max_pre_abs': float(max_pre_abs),
            'horizon_steps': int(horizon_steps),
            'bias_bound': float(bias_bound),
            'estimate': base_est,
            'lower': float(base_est - half),
            'upper': float(base_est + half),
        })

                                                                                 
    linear_bounds = []
    flci_available = len(pre_rows) >= 1 and all('influence' in r and r['influence'] is not None for r in pre_rows + [post_row])
    if flci_available:
        u_pre = np.array([int(r['event_time']) - int(reference_event_time) for r in pre_rows], dtype=float)
        u_post = float(int(target_event_time) - int(reference_event_time))
        u_min = int(min(int(r['event_time']) - int(reference_event_time) for r in pre_rows))
        influences = [np.asarray(r['influence'], dtype=float) for r in pre_rows] + [np.asarray(post_row['influence'], dtype=float)]
        cov = _cluster_covariance_matrix(influences, repository, cve)
        pre_ests = np.array([float(r['estimate']) for r in pre_rows], dtype=float)
        k_pre = len(pre_rows)

        def _flci_bias_coeff(w: np.ndarray) -> float:
            post_term = sum(int(u_post) - r for r in range(1, int(u_post)))
            pre_term = 0.0
            for r in range(u_min + 1, 1):
                s = sum(w[j] * (r - int(u_pre[j])) for j in range(k_pre) if int(u_pre[j]) <= r - 1)
                pre_term += abs(s)
            return float(post_term + pre_term)

        def _flci_se(w: np.ndarray) -> float:
            vec = np.concatenate([-w, [1.0]])
            v = float(vec @ cov @ vec)
            return float(math.sqrt(v)) if v > 0.0 else base_se

                                                               
        w0 = u_post * u_pre / float(np.dot(u_pre, u_pre))
        for m_val in m_values:
                                                                    
            direct_bias = float(m_val)
            direct_half = direct_bias + crit * base_se

                                                                                           
            def _obj(w: np.ndarray, m_cur: float = float(m_val)) -> float:
                return m_cur * _flci_bias_coeff(w) + crit * _flci_se(w)

            cons = {'type': 'eq', 'fun': lambda w: float(np.dot(w, u_pre) - u_post)}
            opt = minimize(_obj, w0, method='SLSQP', constraints=[cons], options={'maxiter': 200, 'ftol': 1e-10})
            w_opt = opt.x if opt.success and np.isfinite(opt.x).all() else w0
            flci_est = float(base_est - np.dot(w_opt, pre_ests))
            flci_bias = float(m_val) * _flci_bias_coeff(w_opt)
            flci_se_val = _flci_se(w_opt)
            flci_half = flci_bias + crit * flci_se_val
            linear_bounds.append({
                'm': float(m_val),
                'estimate': base_est,
                'lower': float(base_est - direct_half),
                'upper': float(base_est + direct_half),
                'flci_estimate': flci_est,
                'flci_lower': float(flci_est - flci_half),
                'flci_upper': float(flci_est + flci_half),
                'flci_bias_bound': flci_bias,
                'flci_se': flci_se_val,
            })
    else:
        for m_val in m_values:
            half = float(m_val) + crit * base_se
            linear_bounds.append({
                'm': float(m_val),
                'estimate': base_est,
                'lower': float(base_est - half),
                'upper': float(base_est + half),
                'flci_estimate': base_est,
                'flci_lower': float(base_est - half),
                'flci_upper': float(base_est + half),
                'flci_bias_bound': float(m_val),
                'flci_se': base_se,
            })

                                                       
    m_breakdown = max(0.0, abs(base_est) - crit * base_se)
    mbar_breakdown = (m_breakdown / (horizon_steps * max_pre_step)) if max_pre_step > 1e-12 else float('inf')

    return {
        'estimable': True,
        'failure_reason': '',
        'target_event_time': int(target_event_time),
        'n_pre_leads': len(pre_rows),
        'max_pre_abs': float(max_pre_abs),
        'max_pre_step': float(max_pre_step),
        'm_breakdown': float(m_breakdown),
        'mbar_breakdown': float(mbar_breakdown) if math.isfinite(mbar_breakdown) else None,
        'linear_bounds': linear_bounds,
        'relative_magnitude_bounds': rm_bounds,
    }


def _coerce_panel_dict(panel_data: dict[str, Any]) -> dict[str, Any]:
    pass
    y = np.asarray(panel_data['y'], dtype=float)
    if y.ndim != 2:
        raise ValueError('panel_data["y"] must be a 2D (n_units, n_weeks) array')
    n, nt = y.shape
    cohort = np.asarray(panel_data['cohort'], dtype=float)
    repository = np.asarray(panel_data['repository'])
    cve = np.asarray(panel_data['cve'])
    if not (len(cohort) == len(repository) == len(cve) == n):
        raise ValueError('Inconsistent unit dimensions across y, cohort, repository, and cve')
    eligibility = (
        np.ones((n, nt), dtype=bool)
        if panel_data.get('eligibility') is None
        else np.asarray(panel_data['eligibility'], dtype=bool)
    )
    observed = (
        np.isfinite(y)
        if panel_data.get('observed') is None
        else (np.asarray(panel_data['observed'], dtype=bool) & np.isfinite(y))
    )
    obs_prob = (
        None
        if panel_data.get('observation_probability') is None
        else np.asarray(panel_data['observation_probability'], dtype=float)
    )
    out = dict(panel_data)
    out.update({
        'y': y,
        'cohort': cohort,
        'repository': repository,
        'cve': cve,
        'eligibility': eligibility,
        'observed': observed,
        'observation_probability': obs_prob,
    })
    return out


def _ols_with_cluster_influence(
    y_vec: np.ndarray,
    design: np.ndarray,
    repository: np.ndarray,
    cve: np.ndarray,
    target_col: int = 0,
) -> dict[str, Any]:
    pass
    n = len(y_vec)
    p = design.shape[1]
    if n <= p:
        return {'valid': False, 'status': 'INSUFFICIENT_CLUSTERS', 'failure_reason': 'Too few observations for OLS design'}
    xtx = design.T @ design
    if np.linalg.cond(xtx) > 1e10:
        return {'valid': False, 'status': 'NOT_IDENTIFIED', 'failure_reason': 'Collinear OLS design matrix'}
    beta = np.linalg.solve(xtx, design.T @ y_vec)
    resid = y_vec - design @ beta
    inv_xtx = np.linalg.inv(xtx)
                                                       
    scores = design * resid[:, None]
    influence = (scores @ inv_xtx)[:, target_col]
    ci = _cluster_interval_from_influence(float(beta[target_col]), influence, repository, cve)
    ci['estimate'] = float(beta[target_col])
    ci['beta'] = beta
    ci['influence'] = influence
    return ci


def run_b0_baselines(
    panel_data: dict[str, Any],
    out_dir: str | Path,
    *,
    config_sha256: str = '',
    input_sha256: str = '',
    run_id: str | None = None,
    mode: str = 'synthetic',
    horizon_days: int = 91,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    anticipation: int = 0,
    mme: float = 0.05,
) -> dict[str, Any]:
    pass
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    panel = _coerce_panel_dict(panel_data)
    lag = int(round(horizon_days / 7.0))
    y = panel['y']
    cohort = panel['cohort']
    repo = panel['repository']
    cve = panel['cve']
    eligibility = panel['eligibility']
    observed = panel['observed']
    obs_prob = panel['observation_probability']

    provenance = _resolve_provenance(
        'B0',
        {'horizon_days': horizon_days, 'lag': lag, 'bootstrap_reps': bootstrap_reps, 'anticipation': anticipation, 'mme': mme},
        {'shape': list(y.shape), 'n_repos': _unique_count(repo), 'n_cves': _unique_count(cve)},
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        run_id=run_id,
        mode=mode,
    )

    did_res = did_aggregate(
        y,
        cohort,
        lag,
        anticipation=anticipation,
        eligibility=eligibility,
        observed=observed,
        observation_probability=obs_prob,
    )
    common_mask = np.asarray(did_res['used'], dtype=bool).copy()
    cvss = panel.get('cvss_score')
    if cvss is not None:
        cvss_arr = np.asarray(cvss, dtype=float)
        common_mask &= np.isfinite(cvss_arr)
    else:
        cvss_arr = None

    if int(common_mask.sum()) < 4:
        raise ValueError('B0 common-support panel has fewer than 4 usable observations')

                                                                   
    idx_common = np.flatnonzero(common_mask)
    y_c = y[idx_common]
    cohort_c = cohort[idx_common]
    repo_c = repo[idx_common]
    cve_c = cve[idx_common]
    elig_c = eligibility[idx_common]
    obs_c = observed[idx_common]
    prob_c = None if obs_prob is None else obs_prob[idx_common]
    cvss_c = None if cvss_arr is None else cvss_arr[idx_common]
    fix_dyad_c = np.asarray(panel['fix_dyad_id'])[idx_common] if panel.get('fix_dyad_id') is not None else None

    raw_keys = panel.get('observation_keys')
    if raw_keys is not None:
        common_keys = [tuple(raw_keys[i]) for i in idx_common]
    else:
        common_keys = [
            (str(repo_c[i]), 'default_project', 'direct_dep', str(cve_c[i]), f'anchor_{int(idx_common[i])}')
            for i in range(len(idx_common))
        ]
    common_keys_sha256 = _hash_json_obj(sorted(common_keys))
    resampling_seed = study_contract.derive_child_seed(study_contract.STUDY_ID, 'B0', 'common_panel_baselines', 0)

    counts = _cluster_counts(repo_c, cve_c, fix_dyad=fix_dyad_c, calendar_shock=cohort_c)
    n_repos = counts['n_repositories']
    n_cves = counts['n_cves']
    n_dyads = counts['n_fix_dyads']
    n_shocks = counts['n_calendar_shocks']

                                                                                   
    finite_groups = sorted(int(g) for g in np.unique(cohort_c) if np.isfinite(g))
    if not finite_groups:
        raise ValueError('B0 common-support panel has no treated cohorts')
    treated_c = np.isfinite(cohort_c)
                                                                                                             
    group_shares = {g: float(np.sum(cohort_c == g)) for g in finite_groups}
    total_treated = sum(group_shares.values())
    post_outcome = np.zeros(len(idx_common), dtype=float)
    base_outcome = np.zeros(len(idx_common), dtype=float)
    eval_cohort = np.zeros(len(idx_common), dtype=int)
    for i in range(len(idx_common)):
        if treated_c[i]:
            g = int(cohort_c[i])
            t_eval = g + lag
            t_base = g - anticipation - 1
            post_outcome[i] = y_c[i, t_eval]
            base_outcome[i] = y_c[i, t_base]
            eval_cohort[i] = g
        else:
                                                                                                   
            val_post = 0.0
            val_base = 0.0
            w_sum = 0.0
            for g in finite_groups:
                t_eval = g + lag
                t_base = g - anticipation - 1
                if obs_c[i, t_eval] and obs_c[i, t_base]:
                    w = group_shares[g] / total_treated
                    val_post += w * y_c[i, t_eval]
                    val_base += w * y_c[i, t_base]
                    w_sum += w
            post_outcome[i] = val_post / w_sum if w_sum > 0 else np.nan
            base_outcome[i] = val_base / w_sum if w_sum > 0 else np.nan
            eval_cohort[i] = finite_groups[0]

    delta_post = post_outcome - base_outcome
    valid_unit = np.isfinite(post_outcome) & np.isfinite(base_outcome)
    weights_unit = np.ones(len(idx_common), dtype=float)
    rows: list[dict[str, Any]] = []
    fairness_models: list[dict[str, Any]] = []

                                                     
    mean_tr, inf_tr = weighted_mean_influence(post_outcome, treated_c & valid_unit, weights_unit)
    mean_ct, inf_ct = weighted_mean_influence(post_outcome, (~treated_c) & valid_unit, weights_unit)
    est_unadj = float(mean_tr - mean_ct)
    ci_unadj = _cluster_interval_from_influence(est_unadj, inf_tr - inf_ct, repo_c, cve_c, seed=resampling_seed)
    row_unadj = _make_result_row(
        experiment_id='B0',
        provenance=provenance,
        estimand='unadjusted_risk_difference',
        exposure='any_treated_cohort',
        comparator='untreated_common_support',
        subgroup='common_support_all',
        horizon_days=horizon_days,
        estimate=est_unadj if ci_unadj['valid'] else None,
        lower=ci_unadj['lower'],
        upper=ci_unadj['upper'],
        scale='risk_difference',
        uncertainty_method='two_way_cluster_influence',
        independent_unit='repository_x_cve',
        n_repositories=n_repos,
        n_cves=n_cves,
        n_fix_dyads=n_dyads,
        n_calendar_shocks=n_shocks,
        multiplicity_family='B0_diagnostic_baselines',
        test_status=ci_unadj['status'],
        failure_reason=ci_unadj['failure_reason'],
    )
    rows.append(row_unadj)
    fairness_models.append({
        'estimand': 'unadjusted_risk_difference',
        'n_units': int(len(idx_common)),
        'horizon_days': int(horizon_days),
        'common_support_keys_sha256': common_keys_sha256,
        'resampling_seed': resampling_seed,
        'test_status': row_unadj['test_status'],
    })

                                                          
    if cvss_c is not None:
        cve_stratum = np.digitize(cvss_c, bins=[4.0, 7.0, 9.0])
    elif panel.get('cve_stratum') is not None:
        cve_stratum = np.asarray(panel['cve_stratum'])[idx_common]
    else:
        cve_stratum = np.zeros(len(idx_common), dtype=int)
    cal_period = np.asarray(panel['calendar_period'])[idx_common] if panel.get('calendar_period') is not None else np.zeros(len(idx_common), dtype=int)
    stratum_ids = np.array([f'{cal_period[i]}:{cve_stratum[i]}' for i in range(len(idx_common))])
    unique_strata = np.unique(stratum_ids)
    matched_est = 0.0
    matched_inf = np.zeros(len(idx_common), dtype=float)
    total_w = 0.0
    for st in unique_strata:
        st_mask = (stratum_ids == st) & valid_unit
        tr_st = st_mask & treated_c
        ct_st = st_mask & (~treated_c)
        if np.sum(tr_st) >= 2 and np.sum(ct_st) >= 2:
            w_st = float(np.sum(st_mask))
            m1, i1 = weighted_mean_influence(post_outcome, tr_st, weights_unit)
            m0, i0 = weighted_mean_influence(post_outcome, ct_st, weights_unit)
            matched_est += w_st * (m1 - m0)
            matched_inf += w_st * (i1 - i0)
            total_w += w_st
    if total_w > 0:
        matched_est /= total_w
        matched_inf /= total_w
        ci_match = _cluster_interval_from_influence(matched_est, matched_inf, repo_c, cve_c, seed=resampling_seed)
        row_match = _make_result_row(
            experiment_id='B0',
            provenance=provenance,
            estimand='calendar_cve_matched_contrast',
            exposure='any_treated_cohort',
            comparator='calendar_cve_stratum_matched',
            subgroup='common_support_all',
            horizon_days=horizon_days,
            estimate=matched_est if ci_match['valid'] else None,
            lower=ci_match['lower'],
            upper=ci_match['upper'],
            scale='risk_difference',
            uncertainty_method='two_way_cluster_influence',
            independent_unit='repository_x_cve',
            n_repositories=n_repos,
            n_cves=n_cves,
            n_fix_dyads=n_dyads,
            n_calendar_shocks=n_shocks,
            multiplicity_family='B0_diagnostic_baselines',
            test_status=ci_match['status'],
            failure_reason=ci_match['failure_reason'],
        )
    else:
        row_match = _nonestimable_row(
            experiment_id='B0',
            provenance=provenance,
            estimand='calendar_cve_matched_contrast',
            exposure='any_treated_cohort',
            comparator='calendar_cve_stratum_matched',
            subgroup='common_support_all',
            horizon_days=horizon_days,
            scale='risk_difference',
            uncertainty_method='two_way_cluster_influence',
            independent_unit='repository_x_cve',
            n_repositories=n_repos,
            n_cves=n_cves,
            n_fix_dyads=n_dyads,
            n_calendar_shocks=n_shocks,
            multiplicity_family='B0_diagnostic_baselines',
            status='NOT_IDENTIFIED',
            failure_reason='No calendar/CVE strata contain both treated and control units on common support',
        )
    rows.append(row_match)
    fairness_models.append({
        'estimand': 'calendar_cve_matched_contrast',
        'n_units': int(len(idx_common)),
        'horizon_days': int(horizon_days),
        'common_support_keys_sha256': common_keys_sha256,
        'resampling_seed': resampling_seed,
        'test_status': row_match['test_status'],
    })

                                                            
    cvss_feature = cvss_c if cvss_c is not None else np.zeros(len(idx_common), dtype=float)
    has_cvss_var = float(np.std(cvss_feature)) > 1e-12
    design_cvss = (
        np.column_stack([treated_c.astype(float), cvss_feature - float(np.mean(cvss_feature)), np.ones(len(idx_common))])
        if has_cvss_var
        else np.column_stack([treated_c.astype(float), np.ones(len(idx_common))])
    )
    ols_cvss = _ols_with_cluster_influence(post_outcome, design_cvss, repo_c, cve_c, target_col=0)
    row_cvss = _make_result_row(
        experiment_id='B0',
        provenance=provenance,
        estimand='static_pretreatment_cvss_contrast',
        exposure='any_treated_cohort',
        comparator='untreated_cvss_adjusted',
        subgroup='common_support_all',
        horizon_days=horizon_days,
        estimate=ols_cvss.get('estimate') if ols_cvss.get('valid') else None,
        lower=ols_cvss.get('lower'),
        upper=ols_cvss.get('upper'),
        scale='risk_difference',
        uncertainty_method='two_way_cluster_influence',
        independent_unit='repository_x_cve',
        n_repositories=n_repos,
        n_cves=n_cves,
        n_fix_dyads=n_dyads,
        n_calendar_shocks=n_shocks,
        multiplicity_family='B0_diagnostic_baselines',
        test_status=ols_cvss['status'],
        failure_reason=ols_cvss.get('failure_reason', '') if ols_cvss.get('valid') else ols_cvss['failure_reason'],
    )
    rows.append(row_cvss)
    fairness_models.append({
        'estimand': 'static_pretreatment_cvss_contrast',
        'n_units': int(len(idx_common)),
        'horizon_days': int(horizon_days),
        'common_support_keys_sha256': common_keys_sha256,
        'resampling_seed': resampling_seed,
        'test_status': row_cvss['test_status'],
    })

                                                                                            
    t_grid = np.arange(y_c.shape[1])[None, :]
    d_panel = (t_grid >= cohort_c[:, None]).astype(float)
    valid_panel = obs_c & elig_c
    n_u, n_t = y_c.shape
    w_sum_u = np.maximum(valid_panel.sum(axis=1, keepdims=True), 1.0)
    w_sum_t = np.maximum(valid_panel.sum(axis=0, keepdims=True), 1.0)
    total_obs_twfe = float(valid_panel.sum())
    if total_obs_twfe >= 4:
                                                                                  
        y_safe = np.where(valid_panel, y_c, 0.0)
        d_safe = np.where(valid_panel, d_panel, 0.0)
        y_tilde = y_safe.copy()
        d_tilde = d_safe.copy()
        for _ in range(15):
            y_tilde = np.where(valid_panel, y_tilde - np.sum(y_tilde, axis=1, keepdims=True) / w_sum_u, 0.0)
            y_tilde = np.where(valid_panel, y_tilde - np.sum(y_tilde, axis=0, keepdims=True) / w_sum_t, 0.0)
            d_tilde = np.where(valid_panel, d_tilde - np.sum(d_tilde, axis=1, keepdims=True) / w_sum_u, 0.0)
            d_tilde = np.where(valid_panel, d_tilde - np.sum(d_tilde, axis=0, keepdims=True) / w_sum_t, 0.0)
        denom_twfe = float(np.sum(d_tilde * d_tilde))
        if denom_twfe > 1e-12:
            beta_twfe = float(np.sum(d_tilde * y_tilde) / denom_twfe)
            resid_twfe = np.where(valid_panel, y_tilde - beta_twfe * d_tilde, 0.0)
            inf_twfe = np.sum(d_tilde * resid_twfe, axis=1) / denom_twfe
            ci_twfe = _cluster_interval_from_influence(beta_twfe, inf_twfe, repo_c, cve_c, seed=resampling_seed)
            twfe_status = 'DIAGNOSTIC_ONLY' if ci_twfe['valid'] else ci_twfe['status']
            twfe_note = 'bias_diagnostic_not_causal' + (f': {ci_twfe["failure_reason"]}' if ci_twfe['failure_reason'] else '')
            row_twfe = _make_result_row(
                experiment_id='B0',
                provenance=provenance,
                estimand='twfe_ols_diagnostic',
                exposure='post_signal_indicator',
                comparator='twfe_unit_and_time_fe_diagnostic',
                subgroup='diagnostic_estimand_noncausal',
                horizon_days=horizon_days,
                estimate=beta_twfe if ci_twfe['valid'] else None,
                lower=ci_twfe['lower'],
                upper=ci_twfe['upper'],
                scale='risk_difference',
                uncertainty_method='two_way_cluster_influence',
                independent_unit='repository_x_cve',
                n_repositories=n_repos,
                n_cves=n_cves,
                n_fix_dyads=n_dyads,
                n_calendar_shocks=n_shocks,
                multiplicity_family='B0_diagnostic_baselines',
                test_status=twfe_status,
                failure_reason=twfe_note,
            )
        else:
            row_twfe = _nonestimable_row(
                experiment_id='B0',
                provenance=provenance,
                estimand='twfe_ols_diagnostic',
                exposure='post_signal_indicator',
                comparator='twfe_unit_and_time_fe_diagnostic',
                subgroup='diagnostic_estimand_noncausal',
                horizon_days=horizon_days,
                scale='risk_difference',
                uncertainty_method='two_way_cluster_influence',
                independent_unit='repository_x_cve',
                n_repositories=n_repos,
                n_cves=n_cves,
                n_fix_dyads=n_dyads,
                n_calendar_shocks=n_shocks,
                multiplicity_family='B0_diagnostic_baselines',
                status='NOT_IDENTIFIED',
                failure_reason='bias_diagnostic_not_causal: zero within-unit treatment variation',
            )
    else:
        row_twfe = _nonestimable_row(
            experiment_id='B0',
            provenance=provenance,
            estimand='twfe_ols_diagnostic',
            exposure='post_signal_indicator',
            comparator='twfe_unit_and_time_fe_diagnostic',
            subgroup='diagnostic_estimand_noncausal',
            horizon_days=horizon_days,
            scale='risk_difference',
            uncertainty_method='two_way_cluster_influence',
            independent_unit='repository_x_cve',
            n_repositories=n_repos,
            n_cves=n_cves,
            n_fix_dyads=n_dyads,
            n_calendar_shocks=n_shocks,
            multiplicity_family='B0_diagnostic_baselines',
            status='NOT_IDENTIFIED',
            failure_reason='bias_diagnostic_not_causal: insufficient panel observations',
        )
    rows.append(row_twfe)
    fairness_models.append({
        'estimand': 'twfe_ols_diagnostic',
        'n_units': int(len(idx_common)),
        'horizon_days': int(horizon_days),
        'common_support_keys_sha256': common_keys_sha256,
        'resampling_seed': resampling_seed,
        'test_status': row_twfe['test_status'],
    })

                                                                 
    did_common = did_aggregate(
        y_c,
        cohort_c,
        lag,
        anticipation=anticipation,
        eligibility=elig_c,
        observed=obs_c,
        observation_probability=prob_c,
    )
    ci_did = _cluster_interval_from_influence(
        did_common['estimate'],
        did_common['influence'],
        repo_c,
        cve_c,
        seed=resampling_seed,
    )
    row_did = _make_result_row(
        experiment_id='B0',
        provenance=provenance,
        estimand='staggered_group_time_did',
        exposure='staggered_cohort_assignment',
        comparator='not_yet_treated_common_support',
        subgroup='common_support_all',
        horizon_days=horizon_days,
        estimate=did_common['estimate'] if ci_did['valid'] else None,
        lower=ci_did['lower'],
        upper=ci_did['upper'],
        scale='risk_difference',
        uncertainty_method='two_way_cluster_influence',
        independent_unit='repository_x_cve',
        n_repositories=n_repos,
        n_cves=n_cves,
        n_fix_dyads=n_dyads,
        n_calendar_shocks=n_shocks,
        multiplicity_family='B0_diagnostic_baselines',
        test_status=ci_did['status'],
        failure_reason=ci_did['failure_reason'],
    )
    rows.append(row_did)
    fairness_models.append({
        'estimand': 'staggered_group_time_did',
        'n_units': int(len(idx_common)),
        'horizon_days': int(horizon_days),
        'common_support_keys_sha256': common_keys_sha256,
        'resampling_seed': resampling_seed,
        'test_status': row_did['test_status'],
    })

    csv_path = out_path / 'common_panel_baselines.csv'
    study_contract.atomic_write_csv(csv_path, study_contract.RESULT_ROW_COLUMNS, rows)

    fairness = {
        'study_id': study_contract.STUDY_ID,
        'experiment_id': 'B0',
        'run_id': provenance['run_id'],
        'common_support_units': int(len(idx_common)),
        'common_support_keys_sha256': common_keys_sha256,
        'n_repositories': n_repos,
        'n_cves': n_cves,
        'n_fix_dyads': n_dyads,
        'n_calendar_shocks': n_shocks,
        'horizon_days': int(horizon_days),
        'resampling_seed': int(resampling_seed),
        'identical_horizons': len({m['horizon_days'] for m in fairness_models}) == 1,
        'identical_common_support': len({(m['n_units'], m['common_support_keys_sha256']) for m in fairness_models}) == 1,
        'identical_resampling_seed': len({m['resampling_seed'] for m in fairness_models}) == 1,
        'twfe_marked_diagnostic_only': (
            row_twfe['test_status'] == 'DIAGNOSTIC_ONLY'
            and 'bias_diagnostic_not_causal' in row_twfe['failure_reason']
        ),
        'models': fairness_models,
    }
    fairness_path = out_path / 'baseline_fairness.json'
    study_contract.atomic_write_json(fairness_path, fairness)
    return {
        'rows': rows,
        'fairness': fairness,
        'artifacts': {
            'common_panel_baselines': str(csv_path),
            'baseline_fairness': str(fairness_path),
        },
    }


def _estimate_did_contrast_row(
    *,
    experiment_id: str,
    provenance: dict[str, str],
    y: np.ndarray,
    cohort: np.ndarray,
    repository: np.ndarray,
    cve: np.ndarray,
    eligibility: np.ndarray,
    observed: np.ndarray,
    observation_probability: np.ndarray | None,
    lag: int,
    horizon_days: int,
    estimand: str,
    exposure: str,
    comparator: str,
    subgroup: str,
    multiplicity_family: str,
    mme: float,
    anticipation: int = 0,
    post_selection: bool = False,
    bootstrap_method: str = 'two_way_influence',
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = study_contract.MASTER_SEED,
    fix_dyad: np.ndarray | None = None,
    calendar_shock: np.ndarray | None = None,
    warning_note: str = '',
    status_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    counts = _cluster_counts(repository, cve, fix_dyad=fix_dyad, calendar_shock=calendar_shock)
    try:
        agg = _did_event_time_aggregate(
            y,
            cohort,
            lag,
            anticipation=anticipation,
            eligibility=eligibility,
            observed=observed,
            observation_probability=observation_probability,
            post_selection=post_selection,
        )
    except ValueError as exc:
        row = _nonestimable_row(
            experiment_id=experiment_id,
            provenance=provenance,
            estimand=estimand,
            exposure=exposure,
            comparator=comparator,
            subgroup=subgroup,
            horizon_days=horizon_days,
            scale='risk_difference',
            uncertainty_method=bootstrap_method,
            independent_unit='repository_x_cve',
            n_repositories=counts['n_repositories'],
            n_cves=counts['n_cves'],
            n_fix_dyads=counts['n_fix_dyads'],
            n_calendar_shocks=counts['n_calendar_shocks'],
            multiplicity_family=multiplicity_family,
            status='NOT_IDENTIFIED',
            failure_reason=str(exc),
        )
        return row, None

    used = agg['used']
    used_counts = _cluster_counts(
        repository[used],
        cve[used],
        fix_dyad=None if fix_dyad is None else fix_dyad[used],
        calendar_shock=None if calendar_shock is None else calendar_shock[used],
    )
    ci = _cluster_interval_from_influence(
        agg['estimate'],
        agg['influence'][used],
        repository[used],
        cve[used],
        bootstrap_method=bootstrap_method,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    status = ci['status']
    if status in study_contract.ESTIMABLE_STATUSES and status_override:
        status = status_override
    reason_parts = [r for r in (ci.get('failure_reason'), warning_note) if r]
    reason = '; '.join(reason_parts)
    mat = _materiality_diagnostics(agg['estimate'] if ci['valid'] else None, ci['lower'], ci['upper'], mme, status)
    row = _make_result_row(
        experiment_id=experiment_id,
        provenance=provenance,
        estimand=estimand,
        exposure=exposure,
        comparator=comparator,
        subgroup=subgroup,
        horizon_days=horizon_days,
        estimate=agg['estimate'] if ci['valid'] else None,
        lower=ci['lower'],
        upper=ci['upper'],
        scale='risk_difference',
        uncertainty_method=bootstrap_method,
        independent_unit='repository_x_cve',
        n_repositories=used_counts['n_repositories'],
        n_cves=used_counts['n_cves'],
        n_fix_dyads=used_counts['n_fix_dyads'],
        n_calendar_shocks=used_counts['n_calendar_shocks'],
        multiplicity_family=multiplicity_family,
        test_status=status,
        failure_reason=reason,
        extra={
            'se': ci.get('se'),
            'p_value': ci.get('p_value'),
            'boot_seed': int(seed),
            **mat,
        },
    )
    return row, agg


def run_e1_epss(
    panel_data: dict[str, Any],
    out_dir: str | Path,
    *,
    config_sha256: str = '',
    input_sha256: str = '',
    run_id: str | None = None,
    mode: str = 'synthetic',
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_method: str = 'two_way_influence',
    mme: float = 0.05,
    anticipation: int = 0,
    event_times: tuple[int, ...] = (-6, -4, -3, -2, -1, 0, 3, 7, 13),
) -> dict[str, Any]:
    pass
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    panel = _coerce_panel_dict(panel_data)
    y = panel['y']
    cohort = panel['cohort']
    repo = panel['repository']
    cve = panel['cve']
    eligibility = panel['eligibility']
    observed = panel['observed']
    obs_prob = panel['observation_probability']
    fix_dyad = np.asarray(panel['fix_dyad_id']) if panel.get('fix_dyad_id') is not None else None
    shock_arr = np.asarray(panel['calendar_shock_id']) if panel.get('calendar_shock_id') is not None else cohort

    provenance = _resolve_provenance(
        'E1',
        {'bootstrap_reps': bootstrap_reps, 'bootstrap_method': bootstrap_method, 'mme': mme, 'anticipation': anticipation},
        {'shape': list(y.shape), 'n_repos': _unique_count(repo), 'n_cves': _unique_count(cve)},
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        run_id=run_id,
        mode=mode,
    )

    finite_shocks = shock_arr[np.isfinite(cohort)]
    if finite_shocks.dtype.kind in 'fcui':
        finite_shocks = finite_shocks[np.isfinite(finite_shocks)]
    n_calendar_shocks = int(len(np.unique(finite_shocks)))
    few_shocks = n_calendar_shocks < E1_MIN_CALENDAR_SHOCKS
    few_shock_msg = (
        f'few_calendar_shocks_warning: n_calendar_shocks={n_calendar_shocks} < {E1_MIN_CALENDAR_SHOCKS}; interpret as association'
        if few_shocks
        else ''
    )

                                                            
    pre_poc = panel.get('pre_poc_present')
    if pre_poc is not None:
        pre_poc_arr = np.asarray(pre_poc, dtype=bool)
        zero_poc_elig = eligibility & (~pre_poc_arr[:, None])
    else:
        pre_poc_arr = None
        zero_poc_elig = eligibility

    seed_primary = study_contract.derive_child_seed(study_contract.STUDY_ID, 'E1', 'primary_day91_cutover', 0)
    effects_rows: list[dict[str, Any]] = []

                                                                                        
    primary_row, primary_agg = _estimate_did_contrast_row(
        experiment_id='E1',
        provenance=provenance,
        y=y,
        cohort=cohort,
        repository=repo,
        cve=cve,
        eligibility=zero_poc_elig,
        observed=observed,
        observation_probability=obs_prob,
        lag=13,
        horizon_days=91,
        estimand='epss_immediate_cutover_day91_active_124',
        exposure='verified_epss_cutover_crossing',
        comparator='never_or_not_yet_crossing_common_support',
        subgroup='zero_pre_poc_direct_deps',
        multiplicity_family='primary_confirmatory',
        mme=mme,
        anticipation=anticipation,
        bootstrap_method=bootstrap_method,
        bootstrap_reps=bootstrap_reps,
        seed=seed_primary,
        fix_dyad=fix_dyad,
        calendar_shock=shock_arr,
        warning_note=few_shock_msg,
    )
    primary_row['few_calendar_shocks_warning'] = few_shocks
    effects_rows.append(primary_row)

                                                                                                   
    alt_boot_method = 'pair_cluster' if bootstrap_method == 'wild_cluster' else 'wild_cluster'
    seed_boot = study_contract.derive_child_seed(study_contract.STUDY_ID, 'E1', f'primary_{alt_boot_method}', 1)
    boot_row, _ = _estimate_did_contrast_row(
        experiment_id='E1',
        provenance=provenance,
        y=y,
        cohort=cohort,
        repository=repo,
        cve=cve,
        eligibility=zero_poc_elig,
        observed=observed,
        observation_probability=obs_prob,
        lag=13,
        horizon_days=91,
        estimand=f'epss_cutover_day91_{alt_boot_method}',
        exposure='verified_epss_cutover_crossing',
        comparator='never_or_not_yet_crossing_common_support',
        subgroup='zero_pre_poc_direct_deps',
        multiplicity_family='E1_uncertainty_sensitivity',
        mme=mme,
        anticipation=anticipation,
        bootstrap_method=alt_boot_method,
        bootstrap_reps=bootstrap_reps,
        seed=seed_boot,
        fix_dyad=fix_dyad,
        calendar_shock=shock_arr,
        warning_note=few_shock_msg,
    )
    boot_row['few_calendar_shocks_warning'] = few_shocks
    effects_rows.append(boot_row)

                                   
    if panel.get('y_day90') is not None:
        y90 = np.asarray(panel['y_day90'], dtype=float)
        obs90 = observed & np.isfinite(y90)
        seed_90 = study_contract.derive_child_seed(study_contract.STUDY_ID, 'E1', 'secondary_day90', 0)
        row_90, _ = _estimate_did_contrast_row(
            experiment_id='E1',
            provenance=provenance,
            y=y90,
            cohort=cohort,
            repository=repo,
            cve=cve,
            eligibility=zero_poc_elig,
            observed=obs90,
            observation_probability=obs_prob,
            lag=13,
            horizon_days=90,
            estimand='epss_immediate_cutover_day90_active_124',
            exposure='verified_epss_cutover_crossing',
            comparator='never_or_not_yet_crossing_common_support',
            subgroup='zero_pre_poc_direct_deps',
            multiplicity_family='E1_secondary',
            mme=mme,
            anticipation=anticipation,
            bootstrap_method=bootstrap_method,
            bootstrap_reps=bootstrap_reps,
            seed=seed_90,
            fix_dyad=fix_dyad,
            calendar_shock=shock_arr,
            warning_note=few_shock_msg,
        )
    else:
        counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
        row_90 = _nonestimable_row(
            experiment_id='E1',
            provenance=provenance,
            estimand='epss_immediate_cutover_day90_active_124',
            exposure='verified_epss_cutover_crossing',
            comparator='never_or_not_yet_crossing_common_support',
            subgroup='zero_pre_poc_direct_deps',
            horizon_days=90,
            scale='risk_difference',
            uncertainty_method=bootstrap_method,
            independent_unit='repository_x_cve',
            n_repositories=counts['n_repositories'],
            n_cves=counts['n_cves'],
            n_fix_dyads=counts['n_fix_dyads'],
            n_calendar_shocks=n_calendar_shocks,
            multiplicity_family='E1_secondary',
            status='NOT_IDENTIFIED',
            failure_reason='Exact day-90 outcome matrix y_day90 absent from panel_data',
        )
    row_90['few_calendar_shocks_warning'] = few_shocks
    effects_rows.append(row_90)

                                       
    if panel.get('y_upgrades_only') is not None:
        y_upg = np.asarray(panel['y_upgrades_only'], dtype=float)
        obs_upg = observed & np.isfinite(y_upg)
        seed_upg = study_contract.derive_child_seed(study_contract.STUDY_ID, 'E1', 'secondary_upgrades_12', 0)
        row_upg, _ = _estimate_did_contrast_row(
            experiment_id='E1',
            provenance=provenance,
            y=y_upg,
            cohort=cohort,
            repository=repo,
            cve=cve,
            eligibility=zero_poc_elig,
            observed=obs_upg,
            observation_probability=obs_prob,
            lag=13,
            horizon_days=91,
            estimand='epss_cutover_day91_upgrades_12',
            exposure='verified_epss_cutover_crossing',
            comparator='never_or_not_yet_crossing_common_support',
            subgroup='zero_pre_poc_direct_deps',
            multiplicity_family='E1_secondary',
            mme=mme,
            anticipation=anticipation,
            bootstrap_method=bootstrap_method,
            bootstrap_reps=bootstrap_reps,
            seed=seed_upg,
            fix_dyad=fix_dyad,
            calendar_shock=shock_arr,
            warning_note=few_shock_msg,
        )
    else:
        counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
        row_upg = _nonestimable_row(
            experiment_id='E1',
            provenance=provenance,
            estimand='epss_cutover_day91_upgrades_12',
            exposure='verified_epss_cutover_crossing',
            comparator='never_or_not_yet_crossing_common_support',
            subgroup='zero_pre_poc_direct_deps',
            horizon_days=91,
            scale='risk_difference',
            uncertainty_method=bootstrap_method,
            independent_unit='repository_x_cve',
            n_repositories=counts['n_repositories'],
            n_cves=counts['n_cves'],
            n_fix_dyads=counts['n_fix_dyads'],
            n_calendar_shocks=n_calendar_shocks,
            multiplicity_family='E1_secondary',
            status='NOT_IDENTIFIED',
            failure_reason='y_upgrades_only outcome absent from panel_data',
        )
    row_upg['few_calendar_shocks_warning'] = few_shocks
    effects_rows.append(row_upg)

                                                     
    if panel.get('y_first_response') is not None:
        y_first = np.asarray(panel['y_first_response'], dtype=float)
        obs_first = observed & np.isfinite(y_first)
        seed_first = study_contract.derive_child_seed(study_contract.STUDY_ID, 'E1', 'secondary_first_response', 0)
        row_first, _ = _estimate_did_contrast_row(
            experiment_id='E1',
            provenance=provenance,
            y=y_first,
            cohort=cohort,
            repository=repo,
            cve=cve,
            eligibility=zero_poc_elig,
            observed=obs_first,
            observation_probability=obs_prob,
            lag=13,
            horizon_days=91,
            estimand='epss_cutover_day91_first_response_active',
            exposure='verified_epss_cutover_crossing',
            comparator='never_or_not_yet_crossing_common_support',
            subgroup='absorbing_first_response',
            multiplicity_family='E1_secondary',
            mme=mme,
            anticipation=anticipation,
            bootstrap_method=bootstrap_method,
            bootstrap_reps=bootstrap_reps,
            seed=seed_first,
            fix_dyad=fix_dyad,
            calendar_shock=shock_arr,
            warning_note=few_shock_msg,
        )
    else:
        counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
        row_first = _nonestimable_row(
            experiment_id='E1',
            provenance=provenance,
            estimand='epss_cutover_day91_first_response_active',
            exposure='verified_epss_cutover_crossing',
            comparator='never_or_not_yet_crossing_common_support',
            subgroup='absorbing_first_response',
            horizon_days=91,
            scale='risk_difference',
            uncertainty_method=bootstrap_method,
            independent_unit='repository_x_cve',
            n_repositories=counts['n_repositories'],
            n_cves=counts['n_cves'],
            n_fix_dyads=counts['n_fix_dyads'],
            n_calendar_shocks=n_calendar_shocks,
            multiplicity_family='E1_secondary',
            status='NOT_IDENTIFIED',
            failure_reason='y_first_response outcome absent from panel_data',
        )
    row_first['few_calendar_shocks_warning'] = few_shocks
    effects_rows.append(row_first)

                                              
    if panel.get('organic_cohort') is not None:
        org_cohort = np.asarray(panel['organic_cohort'], dtype=float)
        seed_org = study_contract.derive_child_seed(study_contract.STUDY_ID, 'E1', 'secondary_organic_crossings', 0)
        row_org, _ = _estimate_did_contrast_row(
            experiment_id='E1',
            provenance=provenance,
            y=y,
            cohort=org_cohort,
            repository=repo,
            cve=cve,
            eligibility=zero_poc_elig,
            observed=observed,
            observation_probability=obs_prob,
            lag=13,
            horizon_days=91,
            estimand='epss_organic_crossings_day91_active_124',
            exposure='organic_score_crossing',
            comparator='never_or_not_yet_organic_crossing',
            subgroup='zero_pre_poc_direct_deps',
            multiplicity_family='E1_secondary',
            mme=mme,
            anticipation=anticipation,
            bootstrap_method=bootstrap_method,
            bootstrap_reps=bootstrap_reps,
            seed=seed_org,
            fix_dyad=fix_dyad,
            calendar_shock=org_cohort,
        )
    else:
        counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
        row_org = _nonestimable_row(
            experiment_id='E1',
            provenance=provenance,
            estimand='epss_organic_crossings_day91_active_124',
            exposure='organic_score_crossing',
            comparator='never_or_not_yet_organic_crossing',
            subgroup='zero_pre_poc_direct_deps',
            horizon_days=91,
            scale='risk_difference',
            uncertainty_method=bootstrap_method,
            independent_unit='repository_x_cve',
            n_repositories=counts['n_repositories'],
            n_cves=counts['n_cves'],
            n_fix_dyads=counts['n_fix_dyads'],
            n_calendar_shocks=0,
            multiplicity_family='E1_secondary',
            status='NOT_IDENTIFIED',
            failure_reason='organic_cohort absent from panel_data',
        )
    row_org['few_calendar_shocks_warning'] = bool(row_org['n_calendar_shocks'] < E1_MIN_CALENDAR_SHOCKS)
    effects_rows.append(row_org)

                                      
    if panel.get('poc_cohort') is not None:
        poc_cohort = np.asarray(panel['poc_cohort'], dtype=float)
        seed_poc = study_contract.derive_child_seed(study_contract.STUDY_ID, 'E1', 'secondary_dated_poc', 0)
        row_poc, _ = _estimate_did_contrast_row(
            experiment_id='E1',
            provenance=provenance,
            y=y,
            cohort=poc_cohort,
            repository=repo,
            cve=cve,
            eligibility=eligibility,
            observed=observed,
            observation_probability=obs_prob,
            lag=13,
            horizon_days=91,
            estimand='dated_poc_publication_day91_active_124',
            exposure='dated_public_poc',
            comparator='not_yet_or_no_poc',
            subgroup='all_direct_deps_dated_poc',
            multiplicity_family='E1_secondary',
            mme=mme,
            anticipation=anticipation,
            bootstrap_method=bootstrap_method,
            bootstrap_reps=bootstrap_reps,
            seed=seed_poc,
            fix_dyad=fix_dyad,
            calendar_shock=poc_cohort,
        )
    else:
        counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
        row_poc = _nonestimable_row(
            experiment_id='E1',
            provenance=provenance,
            estimand='dated_poc_publication_day91_active_124',
            exposure='dated_public_poc',
            comparator='not_yet_or_no_poc',
            subgroup='all_direct_deps_dated_poc',
            horizon_days=91,
            scale='risk_difference',
            uncertainty_method=bootstrap_method,
            independent_unit='repository_x_cve',
            n_repositories=counts['n_repositories'],
            n_cves=counts['n_cves'],
            n_fix_dyads=counts['n_fix_dyads'],
            n_calendar_shocks=0,
            multiplicity_family='E1_secondary',
            status='NOT_IDENTIFIED',
            failure_reason='poc_cohort absent from panel_data',
        )
    row_poc['few_calendar_shocks_warning'] = bool(row_poc['n_calendar_shocks'] < E1_MIN_CALENDAR_SHOCKS)
    effects_rows.append(row_poc)

                                            
    ref_e = -anticipation - 1
    et_rows: list[dict[str, Any]] = []
    honest_input_rows: list[dict[str, Any]] = []
    for et in sorted(set(int(e) for e in event_times)):
        if et == ref_e:
            et_rec = {
                'experiment_id': 'E1',
                'event_time': int(et),
                'horizon_days': int(et * 7),
                'is_reference': True,
                'estimate': 0.0,
                'se': 0.0,
                'lower': 0.0,
                'upper': 0.0,
                'n_treated': int(np.sum(np.isfinite(cohort))),
                'n_controls': int(np.sum(np.isinf(cohort))),
                'test_status': 'ESTIMABLE',
                'failure_reason': 'normalized_reference_period',
            }
            et_rows.append(et_rec)
            continue
        try:
            agg_e = _did_event_time_aggregate(
                y,
                cohort,
                et,
                anticipation=anticipation,
                eligibility=zero_poc_elig,
                observed=observed,
                observation_probability=obs_prob,
            )
            used_e = agg_e['used']
            ci_e = _cluster_interval_from_influence(
                agg_e['estimate'],
                agg_e['influence'][used_e],
                repo[used_e],
                cve[used_e],
                seed=study_contract.derive_child_seed(study_contract.STUDY_ID, 'E1', f'event_time_{et}', 0),
            )
            et_rec = {
                'experiment_id': 'E1',
                'event_time': int(et),
                'horizon_days': int(et * 7),
                'is_reference': False,
                'estimate': float(agg_e['estimate']) if ci_e['valid'] else None,
                'se': ci_e['se'],
                'lower': ci_e['lower'],
                'upper': ci_e['upper'],
                'n_treated': int(sum(c['treated_count'] for c in agg_e['cells'])),
                'n_controls': int(sum(c['control_count'] for c in agg_e['cells'])),
                'test_status': ci_e['status'],
                'failure_reason': ci_e['failure_reason'],
            }
            et_rows.append(et_rec)
            honest_input_rows.append({
                **et_rec,
                'df': ci_e.get('df'),
                'influence': agg_e['influence'],
            })
        except ValueError as exc:
            et_rows.append({
                'experiment_id': 'E1',
                'event_time': int(et),
                'horizon_days': int(et * 7),
                'is_reference': False,
                'estimate': None,
                'se': None,
                'lower': None,
                'upper': None,
                'n_treated': 0,
                'n_controls': 0,
                'test_status': 'NOT_IDENTIFIED',
                'failure_reason': str(exc),
            })

    honest = honest_did_bounds(
        honest_input_rows,
        repo,
        cve,
        target_event_time=13,
        reference_event_time=ref_e,
    )
    if honest['estimable']:
        for lb in honest['linear_bounds']:
            h_row = _make_result_row(
                experiment_id='E1',
                provenance=provenance,
                estimand=f'epss_cutover_honest_did_linear_M_{lb["m"]}',
                exposure='verified_epss_cutover_crossing',
                comparator='never_or_not_yet_crossing_common_support',
                subgroup='zero_pre_poc_direct_deps',
                horizon_days=91,
                estimate=lb['estimate'],
                lower=lb['lower'],
                upper=lb['upper'],
                scale='risk_difference',
                uncertainty_method='honest_did_linear_pretrend_bounds',
                independent_unit='repository_x_cve',
                n_repositories=primary_row['n_repositories'],
                n_cves=primary_row['n_cves'],
                n_fix_dyads=primary_row['n_fix_dyads'],
                n_calendar_shocks=n_calendar_shocks,
                multiplicity_family='E1_honest_did_sensitivity',
                test_status='ESTIMABLE',
                failure_reason=few_shock_msg,
                extra={
                    'honest_did_m': lb['m'],
                    'honest_did_flci_estimate': lb['flci_estimate'],
                    'honest_did_flci_lower': lb['flci_lower'],
                    'honest_did_flci_upper': lb['flci_upper'],
                    'few_calendar_shocks_warning': few_shocks,
                },
            )
            effects_rows.append(h_row)
        for rb in honest['relative_magnitude_bounds']:
            if abs(rb['mbar'] - 1.0) < 1e-12:
                rm_row = _make_result_row(
                    experiment_id='E1',
                    provenance=provenance,
                    estimand='epss_cutover_honest_did_mbar_1',
                    exposure='verified_epss_cutover_crossing',
                    comparator='never_or_not_yet_crossing_common_support',
                    subgroup='zero_pre_poc_direct_deps',
                    horizon_days=91,
                    estimate=rb['estimate'],
                    lower=rb['lower'],
                    upper=rb['upper'],
                    scale='risk_difference',
                    uncertainty_method='honest_did_relative_magnitudes_mbar1',
                    independent_unit='repository_x_cve',
                    n_repositories=primary_row['n_repositories'],
                    n_cves=primary_row['n_cves'],
                    n_fix_dyads=primary_row['n_fix_dyads'],
                    n_calendar_shocks=n_calendar_shocks,
                    multiplicity_family='E1_honest_did_sensitivity',
                    test_status='ESTIMABLE',
                    failure_reason=few_shock_msg,
                    extra={
                        'honest_did_mbar': rb['mbar'],
                        'few_calendar_shocks_warning': few_shocks,
                    },
                )
                effects_rows.append(rm_row)

                                   
    flow_rows: list[dict[str, Any]] = []
    finite_cohorts = sorted(int(g) for g in np.unique(cohort) if np.isfinite(g))
    for g in finite_cohorts + ['never_treated']:
        mask_g = np.isinf(cohort) if g == 'never_treated' else (cohort == float(g))
        n_tot = int(mask_g.sum())
        n_poc_excl = int((mask_g & pre_poc_arr).sum()) if pre_poc_arr is not None else 0
        if g == 'never_treated':
            n_elig = int((mask_g & zero_poc_elig[:, 0]).sum())
            n_obs_sup = int((mask_g & zero_poc_elig[:, 0] & observed[:, 0]).sum())
        else:
            t_eval = int(g) + 13
            t_base = max(0, int(g) - anticipation - 1)
            elig_col = zero_poc_elig[:, max(0, int(g) - 1)]
            obs_pair = (observed[:, t_eval] & observed[:, t_base]) if t_eval < y.shape[1] else np.zeros(len(cohort), dtype=bool)
            n_elig = int((mask_g & elig_col).sum())
            n_obs_sup = int((mask_g & elig_col & obs_pair).sum())
        flow_rows.append({
            'experiment_id': 'E1',
            'cohort_id': str(g),
            'signal_type': 'never_crossing_control' if g == 'never_treated' else 'verified_cutover_shock',
            'n_total': n_tot,
            'n_excluded_pre_poc': n_poc_excl,
            'n_eligible': n_elig,
            'n_observed_support': n_obs_sup,
            'n_unknown_missing': n_elig - n_obs_sup,
            'n_repositories': _unique_count(repo, mask_g),
            'n_cves': _unique_count(cve, mask_g),
            'n_calendar_shocks': n_calendar_shocks,
            'few_calendar_shocks_warning': few_shocks,
        })

    effect_fields = list(study_contract.RESULT_ROW_COLUMNS) + [
        k for k in dict.fromkeys(key for r in effects_rows for key in r) if k not in study_contract.RESULT_ROW_COLUMNS
    ]
    effects_path = out_path / 'epss_effects.csv'
    event_path = out_path / 'epss_event_time.csv'
    flow_path = out_path / 'score_cohort_flow.csv'
    study_contract.atomic_write_csv(effects_path, effect_fields, effects_rows)
    study_contract.atomic_write_csv(event_path, list(et_rows[0].keys()) if et_rows else ['experiment_id', 'event_time'], et_rows)
    study_contract.atomic_write_csv(flow_path, list(flow_rows[0].keys()) if flow_rows else ['experiment_id', 'cohort_id'], flow_rows)

    return {
        'rows': effects_rows,
        'primary_row': primary_row,
        'event_time_rows': et_rows,
        'cohort_flow': flow_rows,
        'honest_did': honest,
        'n_calendar_shocks': n_calendar_shocks,
        'few_calendar_shocks_warning': few_shocks,
        'artifacts': {
            'rq1_effects': str(effects_path),
            'rq1_event_time': str(event_path),
            'score_cohort_flow': str(flow_path),
        },
    }


def run_e2_ghsa_bot(
    panel_data: dict[str, Any],
    out_dir: str | Path,
    *,
    config_sha256: str = '',
    input_sha256: str = '',
    run_id: str | None = None,
    mode: str = 'synthetic',
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_method: str = 'two_way_influence',
    mme: float = 0.04,
    anticipation: int = 0,
    max_pre_weeks: int = 13,
) -> dict[str, Any]:
    pass
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    panel = _coerce_panel_dict(panel_data)
    y = panel['y']
    cohort = panel['cohort']
    repo = panel['repository']
    cve = panel['cve']
    eligibility = panel['eligibility']
    observed = panel['observed']
    obs_prob = panel['observation_probability']
    fix_dyad = np.asarray(panel['fix_dyad_id']) if panel.get('fix_dyad_id') is not None else None
    shock_arr = np.asarray(panel['calendar_shock_id']) if panel.get('calendar_shock_id') is not None else cohort

    provenance = _resolve_provenance(
        'E2',
        {'bootstrap_reps': bootstrap_reps, 'bootstrap_method': bootstrap_method, 'mme': mme, 'max_pre_weeks': max_pre_weeks},
        {'shape': list(y.shape), 'n_repos': _unique_count(repo), 'n_cves': _unique_count(cve)},
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        run_id=run_id,
        mode=mode,
    )

    effects_rows: list[dict[str, Any]] = []
    seed_primary = study_contract.derive_child_seed(study_contract.STUDY_ID, 'E2', 'primary_ghsa_day91', 0)
    primary_row, _ = _estimate_did_contrast_row(
        experiment_id='E2',
        provenance=provenance,
        y=y,
        cohort=cohort,
        repository=repo,
        cve=cve,
        eligibility=eligibility,
        observed=observed,
        observation_probability=obs_prob,
        lag=13,
        horizon_days=91,
        estimand='ghsa_review_day91_active_124',
        exposure='post_fix_ghsa_reviewed',
        comparator='not_yet_reviewed_common_support',
        subgroup='post_fix_ghsa_risk_set',
        multiplicity_family='primary_confirmatory',
        mme=mme,
        anticipation=anticipation,
        bootstrap_method=bootstrap_method,
        bootstrap_reps=bootstrap_reps,
        seed=seed_primary,
        fix_dyad=fix_dyad,
        calendar_shock=shock_arr,
    )
    effects_rows.append(primary_row)

                                        
    if panel.get('y_day90') is not None:
        y90 = np.asarray(panel['y_day90'], dtype=float)
        row_90, _ = _estimate_did_contrast_row(
            experiment_id='E2',
            provenance=provenance,
            y=y90,
            cohort=cohort,
            repository=repo,
            cve=cve,
            eligibility=eligibility,
            observed=observed & np.isfinite(y90),
            observation_probability=obs_prob,
            lag=13,
            horizon_days=90,
            estimand='ghsa_review_day90_active_124',
            exposure='post_fix_ghsa_reviewed',
            comparator='not_yet_reviewed_common_support',
            subgroup='post_fix_ghsa_risk_set',
            multiplicity_family='E2_secondary',
            mme=mme,
            anticipation=anticipation,
            bootstrap_method=bootstrap_method,
            bootstrap_reps=bootstrap_reps,
            seed=study_contract.derive_child_seed(study_contract.STUDY_ID, 'E2', 'secondary_day90', 0),
            fix_dyad=fix_dyad,
            calendar_shock=shock_arr,
        )
        effects_rows.append(row_90)

    if panel.get('y_upgrades_only') is not None:
        y_upg = np.asarray(panel['y_upgrades_only'], dtype=float)
        row_upg, _ = _estimate_did_contrast_row(
            experiment_id='E2',
            provenance=provenance,
            y=y_upg,
            cohort=cohort,
            repository=repo,
            cve=cve,
            eligibility=eligibility,
            observed=observed & np.isfinite(y_upg),
            observation_probability=obs_prob,
            lag=13,
            horizon_days=91,
            estimand='ghsa_review_day91_upgrades_12',
            exposure='post_fix_ghsa_reviewed',
            comparator='not_yet_reviewed_common_support',
            subgroup='post_fix_ghsa_risk_set',
            multiplicity_family='E2_secondary',
            mme=mme,
            anticipation=anticipation,
            bootstrap_method=bootstrap_method,
            bootstrap_reps=bootstrap_reps,
            seed=study_contract.derive_child_seed(study_contract.STUDY_ID, 'E2', 'secondary_upgrades_12', 0),
            fix_dyad=fix_dyad,
            calendar_shock=shock_arr,
        )
        effects_rows.append(row_upg)

                                                                                  
    ref_e = -anticipation - 1
    et_grid = list(range(-int(max_pre_weeks), 14))
    et_rows: list[dict[str, Any]] = []
    honest_input_rows: list[dict[str, Any]] = []
    for et in et_grid:
        if et == ref_e:
            et_rows.append({
                'experiment_id': 'E2',
                'event_time': int(et),
                'horizon_days': int(et * 7),
                'is_reference': True,
                'estimate': 0.0,
                'se': 0.0,
                'lower': 0.0,
                'upper': 0.0,
                'n_treated': int(np.sum(np.isfinite(cohort))),
                'n_controls': int(np.sum(np.isinf(cohort))),
                'test_status': 'ESTIMABLE',
                'failure_reason': 'normalized_reference_period',
            })
            continue
        try:
            agg_e = _did_event_time_aggregate(
                y,
                cohort,
                et,
                anticipation=anticipation,
                eligibility=eligibility,
                observed=observed,
                observation_probability=obs_prob,
            )
            used_e = agg_e['used']
            ci_e = _cluster_interval_from_influence(
                agg_e['estimate'],
                agg_e['influence'][used_e],
                repo[used_e],
                cve[used_e],
                seed=study_contract.derive_child_seed(study_contract.STUDY_ID, 'E2', f'event_time_{et}', 0),
            )
            et_rec = {
                'experiment_id': 'E2',
                'event_time': int(et),
                'horizon_days': int(et * 7),
                'is_reference': False,
                'estimate': float(agg_e['estimate']) if ci_e['valid'] else None,
                'se': ci_e['se'],
                'lower': ci_e['lower'],
                'upper': ci_e['upper'],
                'n_treated': int(sum(c['treated_count'] for c in agg_e['cells'])),
                'n_controls': int(sum(c['control_count'] for c in agg_e['cells'])),
                'test_status': ci_e['status'],
                'failure_reason': ci_e['failure_reason'],
            }
            et_rows.append(et_rec)
            honest_input_rows.append({**et_rec, 'df': ci_e.get('df'), 'influence': agg_e['influence']})
        except ValueError as exc:
            et_rows.append({
                'experiment_id': 'E2',
                'event_time': int(et),
                'horizon_days': int(et * 7),
                'is_reference': False,
                'estimate': None,
                'se': None,
                'lower': None,
                'upper': None,
                'n_treated': 0,
                'n_controls': 0,
                'test_status': 'NOT_IDENTIFIED',
                'failure_reason': str(exc),
            })

    honest = honest_did_bounds(
        honest_input_rows,
        repo,
        cve,
        target_event_time=13,
        reference_event_time=ref_e,
    )
    if honest['estimable']:
        for lb in honest['linear_bounds']:
            effects_rows.append(_make_result_row(
                experiment_id='E2',
                provenance=provenance,
                estimand=f'ghsa_honest_did_linear_M_{lb["m"]}',
                exposure='post_fix_ghsa_reviewed',
                comparator='not_yet_reviewed_common_support',
                subgroup='post_fix_ghsa_risk_set',
                horizon_days=91,
                estimate=lb['estimate'],
                lower=lb['lower'],
                upper=lb['upper'],
                scale='risk_difference',
                uncertainty_method='honest_did_linear_pretrend_bounds',
                independent_unit='repository_x_cve',
                n_repositories=primary_row['n_repositories'],
                n_cves=primary_row['n_cves'],
                n_fix_dyads=primary_row['n_fix_dyads'],
                n_calendar_shocks=primary_row['n_calendar_shocks'],
                multiplicity_family='E2_honest_did_sensitivity',
                test_status='ESTIMABLE',
                failure_reason='',
                extra={
                    'honest_did_m': lb['m'],
                    'honest_did_flci_estimate': lb['flci_estimate'],
                    'honest_did_flci_lower': lb['flci_lower'],
                    'honest_did_flci_upper': lb['flci_upper'],
                },
            ))
        for rb in honest['relative_magnitude_bounds']:
            if abs(rb['mbar'] - 1.0) < 1e-12:
                effects_rows.append(_make_result_row(
                    experiment_id='E2',
                    provenance=provenance,
                    estimand='ghsa_honest_did_mbar_1',
                    exposure='post_fix_ghsa_reviewed',
                    comparator='not_yet_reviewed_common_support',
                    subgroup='post_fix_ghsa_risk_set',
                    horizon_days=91,
                    estimate=rb['estimate'],
                    lower=rb['lower'],
                    upper=rb['upper'],
                    scale='risk_difference',
                    uncertainty_method='honest_did_relative_magnitudes_mbar1',
                    independent_unit='repository_x_cve',
                    n_repositories=primary_row['n_repositories'],
                    n_cves=primary_row['n_cves'],
                    n_fix_dyads=primary_row['n_fix_dyads'],
                    n_calendar_shocks=primary_row['n_calendar_shocks'],
                    multiplicity_family='E2_honest_did_sensitivity',
                    test_status='ESTIMABLE',
                    failure_reason='',
                    extra={'honest_did_mbar': rb['mbar']},
                ))

                                                                                 
    bot_rows: list[dict[str, Any]] = []
    pre_bot = panel.get('pre_signal_bot')
    if pre_bot is not None:
        bot_arr = np.asarray(pre_bot, dtype=float)
        known_bot = np.isfinite(bot_arr) & ((bot_arr == 0.0) | (bot_arr == 1.0))
        if panel.get('bot_head_only') is True:
            counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
            bot_rows.append(_nonestimable_row(
                experiment_id='E2',
                provenance=provenance,
                estimand='ghsa_bot_interaction_contrast',
                exposure='pre_signal_bot_present_vs_absent',
                comparator='bot_absent_common_support',
                subgroup='pre_signal_bot_moderation',
                horizon_days=91,
                scale='risk_difference',
                uncertainty_method=bootstrap_method,
                independent_unit='repository_x_cve',
                n_repositories=counts['n_repositories'],
                n_cves=counts['n_cves'],
                n_fix_dyads=counts['n_fix_dyads'],
                n_calendar_shocks=counts['n_calendar_shocks'],
                multiplicity_family='E2_secondary',
                status='BLOCKED',
                failure_reason='FC-03 violation: current HEAD bot state cannot substitute for pre-signal bot snapshot',
            ))
        else:
                                                                                   
            support_mask = known_bot.copy()
            if panel.get('bot_propensity') is not None:
                ps = np.asarray(panel['bot_propensity'], dtype=float)
                support_mask &= np.isfinite(ps) & (ps >= 0.05) & (ps <= 0.95)
            bot_aggs: dict[int, dict[str, Any]] = {}
            for bot_val, label in ((1, 'pre_signal_bot_present'), (0, 'pre_signal_bot_absent')):
                stratum_mask = support_mask & (bot_arr == float(bot_val))
                elig_stratum = eligibility & stratum_mask[:, None]
                row_b, agg_b = _estimate_did_contrast_row(
                    experiment_id='E2',
                    provenance=provenance,
                    y=y,
                    cohort=cohort,
                    repository=repo,
                    cve=cve,
                    eligibility=elig_stratum,
                    observed=observed,
                    observation_probability=obs_prob,
                    lag=13,
                    horizon_days=91,
                    estimand=f'ghsa_day91_{label}',
                    exposure='post_fix_ghsa_reviewed',
                    comparator='not_yet_reviewed_common_support',
                    subgroup=label,
                    multiplicity_family='E2_secondary',
                    mme=mme,
                    anticipation=anticipation,
                    bootstrap_method=bootstrap_method,
                    bootstrap_reps=bootstrap_reps,
                    seed=study_contract.derive_child_seed(study_contract.STUDY_ID, 'E2', label, 0),
                    fix_dyad=fix_dyad,
                    calendar_shock=shock_arr,
                )
                bot_rows.append(row_b)
                if agg_b is not None and row_b['test_status'] in study_contract.ESTIMABLE_STATUSES:
                    bot_aggs[bot_val] = agg_b
            if 1 in bot_aggs and 0 in bot_aggs:
                diff_est = float(bot_aggs[1]['estimate'] - bot_aggs[0]['estimate'])
                diff_inf = bot_aggs[1]['influence'] - bot_aggs[0]['influence']
                used_both = bot_aggs[1]['used'] | bot_aggs[0]['used']
                cnt_both = _cluster_counts(
                    repo[used_both],
                    cve[used_both],
                    fix_dyad=None if fix_dyad is None else fix_dyad[used_both],
                    calendar_shock=shock_arr[used_both],
                )
                ci_int = _cluster_interval_from_influence(
                    diff_est,
                    diff_inf[used_both],
                    repo[used_both],
                    cve[used_both],
                    bootstrap_method=bootstrap_method,
                    bootstrap_reps=bootstrap_reps,
                    seed=study_contract.derive_child_seed(study_contract.STUDY_ID, 'E2', 'bot_interaction', 0),
                )
                mat_int = _materiality_diagnostics(
                    diff_est if ci_int['valid'] else None,
                    ci_int['lower'],
                    ci_int['upper'],
                    mme,
                    ci_int['status'],
                )
                bot_rows.append(_make_result_row(
                    experiment_id='E2',
                    provenance=provenance,
                    estimand='ghsa_bot_interaction_contrast',
                    exposure='pre_signal_bot_present_minus_absent',
                    comparator='common_support_within_bot_strata',
                    subgroup='pre_signal_bot_interaction',
                    horizon_days=91,
                    estimate=diff_est if ci_int['valid'] else None,
                    lower=ci_int['lower'],
                    upper=ci_int['upper'],
                    scale='risk_difference_interaction',
                    uncertainty_method=bootstrap_method,
                    independent_unit='repository_x_cve',
                    n_repositories=cnt_both['n_repositories'],
                    n_cves=cnt_both['n_cves'],
                    n_fix_dyads=cnt_both['n_fix_dyads'],
                    n_calendar_shocks=cnt_both['n_calendar_shocks'],
                    multiplicity_family='E2_secondary',
                    test_status=ci_int['status'],
                    failure_reason=ci_int['failure_reason'],
                    extra={'se': ci_int.get('se'), 'p_value': ci_int.get('p_value'), **mat_int},
                ))
            else:
                counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
                bot_rows.append(_nonestimable_row(
                    experiment_id='E2',
                    provenance=provenance,
                    estimand='ghsa_bot_interaction_contrast',
                    exposure='pre_signal_bot_present_minus_absent',
                    comparator='common_support_within_bot_strata',
                    subgroup='pre_signal_bot_interaction',
                    horizon_days=91,
                    scale='risk_difference_interaction',
                    uncertainty_method=bootstrap_method,
                    independent_unit='repository_x_cve',
                    n_repositories=counts['n_repositories'],
                    n_cves=counts['n_cves'],
                    n_fix_dyads=counts['n_fix_dyads'],
                    n_calendar_shocks=counts['n_calendar_shocks'],
                    multiplicity_family='E2_secondary',
                    status='NOT_IDENTIFIED',
                    failure_reason='Common support missing in bot_present or bot_absent stratum',
                ))
    else:
        counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
        bot_rows.append(_nonestimable_row(
            experiment_id='E2',
            provenance=provenance,
            estimand='ghsa_bot_interaction_contrast',
            exposure='pre_signal_bot_present_minus_absent',
            comparator='common_support_within_bot_strata',
            subgroup='pre_signal_bot_interaction',
            horizon_days=91,
            scale='risk_difference_interaction',
            uncertainty_method=bootstrap_method,
            independent_unit='repository_x_cve',
            n_repositories=counts['n_repositories'],
            n_cves=counts['n_cves'],
            n_fix_dyads=counts['n_fix_dyads'],
            n_calendar_shocks=counts['n_calendar_shocks'],
            multiplicity_family='E2_secondary',
            status='NOT_IDENTIFIED',
            failure_reason='pre_signal_bot column absent from panel_data',
        ))

    effect_fields = list(study_contract.RESULT_ROW_COLUMNS) + [
        k for k in dict.fromkeys(key for r in effects_rows for key in r) if k not in study_contract.RESULT_ROW_COLUMNS
    ]
    bot_fields = list(study_contract.RESULT_ROW_COLUMNS) + [
        k for k in dict.fromkeys(key for r in bot_rows for key in r) if k not in study_contract.RESULT_ROW_COLUMNS
    ]
    effects_path = out_path / 'advisory_effects.csv'
    event_path = out_path / 'advisory_event_time.csv'
    bot_path = out_path / 'bot_moderation.csv'
    study_contract.atomic_write_csv(effects_path, effect_fields, effects_rows)
    study_contract.atomic_write_csv(event_path, list(et_rows[0].keys()) if et_rows else ['experiment_id', 'event_time'], et_rows)
    study_contract.atomic_write_csv(bot_path, bot_fields, bot_rows)

    return {
        'rows': effects_rows,
        'primary_row': primary_row,
        'event_time_rows': et_rows,
        'bot_moderation_rows': bot_rows,
        'honest_did': honest,
        'artifacts': {
            'ghsa_effects': str(effects_path),
            'ghsa_event_time': str(event_path),
            'bot_moderation': str(bot_path),
        },
    }


def _matched_block_permutation_test(
    y: np.ndarray,
    cohort: np.ndarray,
    cve: np.ndarray,
    ecosystem: np.ndarray,
    matched_block: np.ndarray,
    eligibility: np.ndarray,
    observed: np.ndarray,
    *,
    lag: int = 13,
    anticipation: int = 0,
    perm_reps: int = DEFAULT_PERM_REPS,
    seed: int = study_contract.MASTER_SEED,
    differential: bool = True,
) -> dict[str, Any]:
    pass
                                                                                       
    finite_groups = sorted(int(g) for g in np.unique(cohort) if np.isfinite(g))
    if not finite_groups or int(perm_reps) < 1:
        return {'valid': False, 'p_value': None, 'exceed': 0, 'perm_reps': int(perm_reps), 't_obs': None}

                                                           
                                                                                                                        
    unique_cves, cve_inv = np.unique(cve, return_inverse=True)
    n_cves = len(unique_cves)
    cve_cohort = np.full(n_cves, np.inf, dtype=float)
    cve_eco = np.empty(n_cves, dtype=object)
    cve_block = np.empty(n_cves, dtype=object)
    cve_delta = np.zeros(n_cves, dtype=float)
    cve_weight = np.zeros(n_cves, dtype=float)

    ref_weights = {g: float(np.sum(cohort == float(g))) for g in finite_groups}
    tot_ref_w = sum(ref_weights.values())
    for idx_c in range(n_cves):
        m = cve_inv == idx_c
        eco_vals = ecosystem[m]
        cve_eco[idx_c] = str(eco_vals[0]).lower()
        cve_block[idx_c] = str(matched_block[m][0])
        c_vals = cohort[m]
        fin = c_vals[np.isfinite(c_vals)]
        if len(fin) > 0:
            g_use = int(fin[0])
            cve_cohort[idx_c] = float(g_use)
            t_eval = g_use + lag
            t_base = g_use - anticipation - 1
            if 0 <= t_base < y.shape[1] and 0 <= t_eval < y.shape[1]:
                ok = m & eligibility[:, max(0, g_use - 1)] & observed[:, t_eval] & observed[:, t_base]
                if np.any(ok):
                    cve_delta[idx_c] = float(np.mean(y[ok, t_eval] - y[ok, t_base]))
                    cve_weight[idx_c] = float(np.sum(ok))
        else:
            weighted_deltas = []
            w_list = []
            cnt_max = 0
            for g_use in finite_groups:
                t_eval = g_use + lag
                t_base = g_use - anticipation - 1
                if 0 <= t_base < y.shape[1] and 0 <= t_eval < y.shape[1]:
                    ok = m & eligibility[:, max(0, g_use - 1)] & observed[:, t_eval] & observed[:, t_base]
                    if np.any(ok):
                        weighted_deltas.append(float(np.mean(y[ok, t_eval] - y[ok, t_base])))
                        w_list.append(ref_weights[g_use] / tot_ref_w)
                        cnt_max = max(cnt_max, int(np.sum(ok)))
            if weighted_deltas:
                cve_delta[idx_c] = float(np.average(weighted_deltas, weights=w_list))
                cve_weight[idx_c] = float(cnt_max)

    valid_c = cve_weight > 0
    if int(valid_c.sum()) < 4:
        return {'valid': False, 'p_value': None, 'exceed': 0, 'perm_reps': int(perm_reps), 't_obs': None}

    treated_flag = np.isfinite(cve_cohort)
    is_maven = cve_eco == 'maven'
    is_npm_pypi = (cve_eco == 'npm') | (cve_eco == 'pypi')

    def _stat(tr_vec: np.ndarray) -> float:
        if differential:
            m1 = valid_c & is_maven & tr_vec
            m0 = valid_c & is_maven & (~tr_vec)
            p1 = valid_c & is_npm_pypi & tr_vec
            p0 = valid_c & is_npm_pypi & (~tr_vec)
            if not (np.any(m1) and np.any(m0) and np.any(p1) and np.any(p0)):
                return np.nan
            att_m = np.average(cve_delta[m1], weights=cve_weight[m1]) - np.average(cve_delta[m0], weights=cve_weight[m0])
            att_p = np.average(cve_delta[p1], weights=cve_weight[p1]) - np.average(cve_delta[p0], weights=cve_weight[p0])
            return float(att_m - att_p)
        t1 = valid_c & tr_vec
        t0 = valid_c & (~tr_vec)
        if not (np.any(t1) and np.any(t0)):
            return np.nan
        return float(np.average(cve_delta[t1], weights=cve_weight[t1]) - np.average(cve_delta[t0], weights=cve_weight[t0]))

    t_obs = _stat(treated_flag)
    if not math.isfinite(t_obs):
        return {'valid': False, 'p_value': None, 'exceed': 0, 'perm_reps': int(perm_reps), 't_obs': None}

                                                                     
    block_keys = np.array([f'{cve_eco[i]}|{cve_block[i]}' if valid_c[i] else '__invalid__' for i in range(n_cves)])
    perm_groups: list[np.ndarray] = []
    for bk in np.unique(block_keys):
        if bk == '__invalid__':
            continue
        idxs = np.flatnonzero(block_keys == bk)
        if len(idxs) >= 2 and 0 < int(np.sum(treated_flag[idxs])) < len(idxs):
            perm_groups.append(idxs)

    if not perm_groups:
        return {'valid': False, 'p_value': None, 'exceed': 0, 'perm_reps': int(perm_reps), 't_obs': float(t_obs)}

    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(int(perm_reps)):
        perm_tr = treated_flag.copy()
        for idxs in perm_groups:
            perm_tr[idxs] = rng.permutation(treated_flag[idxs])
        t_rep = _stat(perm_tr)
        if math.isfinite(t_rep) and abs(t_rep) >= abs(t_obs) - 1e-12:
            exceed += 1

    p_val = float((exceed + 1) / (int(perm_reps) + 1))
    return {
        'valid': True,
        'p_value': p_val,
        'exceed': int(exceed),
        'perm_reps': int(perm_reps),
        't_obs': float(t_obs),
        'n_blocks': len(perm_groups),
        'seed': int(seed),
    }


def run_e3_kev_ecosystem(
    panel_data: dict[str, Any],
    out_dir: str | Path,
    *,
    config_sha256: str = '',
    input_sha256: str = '',
    run_id: str | None = None,
    mode: str = 'synthetic',
    perm_reps: int = DEFAULT_PERM_REPS,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    mme: float = 0.04,
    anticipation: int = 0,
) -> dict[str, Any]:
    pass
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    panel = _coerce_panel_dict(panel_data)
    y = panel['y']
    cohort = panel['cohort']
    repo = panel['repository']
    cve = panel['cve']
    eligibility = panel['eligibility']
    observed = panel['observed']
    obs_prob = panel['observation_probability']
    fix_dyad = np.asarray(panel['fix_dyad_id']) if panel.get('fix_dyad_id') is not None else None
    shock_arr = np.asarray(panel['calendar_shock_id']) if panel.get('calendar_shock_id') is not None else cohort

    n_units = len(cohort)
    if panel.get('ecosystem') is None:
        raise ValueError('E3 requires panel_data["ecosystem"]')
    ecosystem = np.array([str(e).strip().lower() for e in panel['ecosystem']], dtype=object)
    matched_block = (
        np.asarray(panel['matched_block_id'], dtype=object)
        if panel.get('matched_block_id') is not None
        else np.zeros(n_units, dtype=object)
    )

    provenance = _resolve_provenance(
        'E3',
        {'perm_reps': perm_reps, 'bootstrap_reps': bootstrap_reps, 'mme': mme},
        {'shape': list(y.shape), 'n_repos': _unique_count(repo), 'n_cves': _unique_count(cve)},
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        run_id=run_id,
        mode=mode,
    )

    def _exposed_cve_count(mask: np.ndarray) -> int:
        exp_mask = mask & np.isfinite(cohort)
        return _unique_count(cve, exp_mask)

    maven_mask = ecosystem == 'maven'
    npm_pypi_mask = (ecosystem == 'npm') | (ecosystem == 'pypi')
    all_mask = np.ones(n_units, dtype=bool)

    effects_rows: list[dict[str, Any]] = []
    cluster_count_rows: list[dict[str, Any]] = []

    def _eval_kev_stratum(
        mask: np.ndarray,
        estimand: str,
        subgroup: str,
        multiplicity_family: str,
        seed_label: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        n_exp_cve = _exposed_cve_count(mask)
        n_ctrl_cve = _unique_count(cve, mask & np.isinf(cohort))
        n_repo_stratum = _unique_count(repo, mask)
        few_clust = n_exp_cve < E3_MIN_EXPOSED_CVE_CLUSTERS
        cluster_count_rows.append({
            'experiment_id': 'E3',
            'stratum': subgroup,
            'n_exposed_cve_clusters': n_exp_cve,
            'n_unexposed_cve_clusters': n_ctrl_cve,
            'n_total_cve_clusters': _unique_count(cve, mask),
            'n_repositories': n_repo_stratum,
            'n_observations': int(mask.sum()),
            'below_25_exposed_threshold': few_clust,
        })
        status_ovr = 'LIMITED_INFERENCE_FEW_CLUSTERS' if (2 <= n_exp_cve < E3_MIN_EXPOSED_CVE_CLUSTERS) else None
        warn = (
            f'few_exposed_cve_clusters_lt_{E3_MIN_EXPOSED_CVE_CLUSTERS}: n_exposed_cves={n_exp_cve}'
            if few_clust
            else ''
        )
        row, agg = _estimate_did_contrast_row(
            experiment_id='E3',
            provenance=provenance,
            y=y,
            cohort=cohort,
            repository=repo,
            cve=cve,
            eligibility=eligibility & mask[:, None],
            observed=observed,
            observation_probability=obs_prob,
            lag=13,
            horizon_days=91,
            estimand=estimand,
            exposure='post_fix_kev_listed',
            comparator='not_yet_kev_listed_common_support',
            subgroup=subgroup,
            multiplicity_family=multiplicity_family,
            mme=mme,
            anticipation=anticipation,
            seed=study_contract.derive_child_seed(study_contract.STUDY_ID, 'E3', seed_label, 0),
            fix_dyad=fix_dyad,
            calendar_shock=shock_arr,
            warning_note=warn,
            status_override=status_ovr,
        )
        row['n_exposed_cve_clusters'] = n_exp_cve
        return row, agg

    row_maven, agg_maven = _eval_kev_stratum(maven_mask, 'kev_day91_maven_active_124', 'maven', 'E3_secondary', 'maven')
    row_np, agg_np = _eval_kev_stratum(npm_pypi_mask, 'kev_day91_npm_pypi_active_124', 'npm_pypi', 'E3_secondary', 'npm_pypi')
    row_pool, agg_pool = _eval_kev_stratum(all_mask, 'kev_day91_pooled_active_124', 'pooled_all_ecosystems', 'E3_secondary', 'pooled')

                                                                         
    n_exp_maven = _exposed_cve_count(maven_mask)
    n_exp_np = _exposed_cve_count(npm_pypi_mask)
    n_exp_total = _exposed_cve_count(all_mask)
    diff_few_clust = (n_exp_total < E3_MIN_EXPOSED_CVE_CLUSTERS) or (min(n_exp_maven, n_exp_np) < E3_MIN_EXPOSED_CVE_CLUSTERS)
    cluster_count_rows.append({
        'experiment_id': 'E3',
        'stratum': 'maven_minus_npm_pypi',
        'n_exposed_cve_clusters': n_exp_total,
        'n_unexposed_cve_clusters': _unique_count(cve, np.isinf(cohort)),
        'n_total_cve_clusters': _unique_count(cve),
        'n_repositories': _unique_count(repo),
        'n_observations': int(n_units),
        'below_25_exposed_threshold': diff_few_clust,
    })

    perm_seed = study_contract.derive_child_seed(study_contract.STUDY_ID, 'E3', 'matched_block_permutation', 0)
    perm_res = _matched_block_permutation_test(
        y,
        cohort,
        cve,
        ecosystem,
        matched_block,
        eligibility,
        observed,
        lag=13,
        anticipation=anticipation,
        perm_reps=perm_reps,
        seed=perm_seed,
        differential=True,
    )

    if agg_maven is not None and agg_np is not None and row_maven['test_status'] in study_contract.ESTIMABLE_STATUSES and row_np['test_status'] in study_contract.ESTIMABLE_STATUSES:
        diff_est = float(agg_maven['estimate'] - agg_np['estimate'])
        diff_inf = agg_maven['influence'] - agg_np['influence']
        used_both = agg_maven['used'] | agg_np['used']
        cnt_diff = _cluster_counts(
            repo[used_both],
            cve[used_both],
            fix_dyad=None if fix_dyad is None else fix_dyad[used_both],
            calendar_shock=shock_arr[used_both],
        )
        ci_diff = _cluster_interval_from_influence(
            diff_est,
            diff_inf[used_both],
            repo[used_both],
            cve[used_both],
            seed=study_contract.derive_child_seed(study_contract.STUDY_ID, 'E3', 'maven_minus_npm_pypi', 0),
            df_cves_override=max(2, min(n_exp_maven, n_exp_np)),
        )
        diff_status = ci_diff['status']
        if diff_status in study_contract.ESTIMABLE_STATUSES and diff_few_clust:
            diff_status = 'LIMITED_INFERENCE_FEW_CLUSTERS'
        diff_reason = ci_diff['failure_reason']
        if diff_few_clust:
            msg = (
                f'few_exposed_cve_clusters_lt_{E3_MIN_EXPOSED_CVE_CLUSTERS}: '
                f'total_exposed={n_exp_total}, maven_exposed={n_exp_maven}, npm_pypi_exposed={n_exp_np}; '
                'exchangeability_dependent_permutation_sensitivity'
            )
            diff_reason = f'{diff_reason}; {msg}' if diff_reason else msg
        mat_diff = _materiality_diagnostics(diff_est if ci_diff['valid'] else None, ci_diff['lower'], ci_diff['upper'], mme, diff_status)
        row_diff = _make_result_row(
            experiment_id='E3',
            provenance=provenance,
            estimand='kev_day91_maven_minus_npm_pypi_differential',
            exposure='kev_x_maven',
            comparator='kev_x_npm_or_pypi',
            subgroup='maven_minus_npm_pypi',
            horizon_days=91,
            estimate=diff_est if ci_diff['valid'] else None,
            lower=ci_diff['lower'],
            upper=ci_diff['upper'],
            scale='risk_difference_interaction',
            uncertainty_method='two_way_cluster_influence_plus_matched_block_perm',
            independent_unit='exposed_cve_and_repository_clusters',
            n_repositories=cnt_diff['n_repositories'],
            n_cves=cnt_diff['n_cves'],
            n_fix_dyads=cnt_diff['n_fix_dyads'],
            n_calendar_shocks=cnt_diff['n_calendar_shocks'],
            multiplicity_family='primary_confirmatory',
            test_status=diff_status,
            failure_reason=diff_reason,
            extra={
                'se': ci_diff.get('se'),
                'p_value': ci_diff.get('p_value'),
                'n_exposed_cve_clusters': n_exp_total,
                'n_exposed_maven_cves': n_exp_maven,
                'n_exposed_npm_pypi_cves': n_exp_np,
                'perm_p_value': perm_res.get('p_value'),
                'perm_exceed': perm_res.get('exceed'),
                'perm_reps': perm_res.get('perm_reps'),
                'perm_seed': perm_seed,
                'perm_assumption': 'exchangeability_within_matched_blocks_not_randomized',
                **mat_diff,
            },
        )
    else:
        counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
        row_diff = _nonestimable_row(
            experiment_id='E3',
            provenance=provenance,
            estimand='kev_day91_maven_minus_npm_pypi_differential',
            exposure='kev_x_maven',
            comparator='kev_x_npm_or_pypi',
            subgroup='maven_minus_npm_pypi',
            horizon_days=91,
            scale='risk_difference_interaction',
            uncertainty_method='two_way_cluster_influence',
            independent_unit='exposed_cve_and_repository_clusters',
            n_repositories=counts['n_repositories'],
            n_cves=counts['n_cves'],
            n_fix_dyads=counts['n_fix_dyads'],
            n_calendar_shocks=counts['n_calendar_shocks'],
            multiplicity_family='primary_confirmatory',
            status='INSUFFICIENT_CLUSTERS',
            failure_reason='Insufficient clusters in Maven or npm/PyPI stratum for differential',
            extra={'n_exposed_cve_clusters': n_exp_total, 'perm_p_value': perm_res.get('p_value'), 'perm_reps': perm_res.get('perm_reps')},
        )

    effects_rows.extend([row_diff, row_pool, row_maven, row_np])

                                               
    regime_rows: list[dict[str, Any]] = []
    regimes_raw = panel.get('policy_regime')
    if regimes_raw is not None:
        regimes = np.array([str(r) for r in regimes_raw], dtype=object)
        regime_verified = panel.get('policy_regime_verified')
        for reg_name in sorted(set(regimes.tolist())):
            reg_mask = regimes == reg_name
            is_unverified = ('unverified' in reg_name.lower()) or (
                regime_verified is not None and not bool(np.all(np.asarray(regime_verified)[reg_mask]))
            )
            n_exp_reg = _exposed_cve_count(reg_mask)
            status_reg = 'LIMITED_INFERENCE_FEW_CLUSTERS' if (2 <= n_exp_reg < E3_MIN_EXPOSED_CVE_CLUSTERS) else None
            note_reg = (
                'unverified_policy_duty_mandate_wording_blocked_catalog_association_only'
                if is_unverified
                else 'verified_policy_regime_stratum'
            )
            if n_exp_reg < E3_MIN_EXPOSED_CVE_CLUSTERS:
                note_reg += f'; few_exposed_cve_clusters_lt_{E3_MIN_EXPOSED_CVE_CLUSTERS}'
            row_reg, _ = _estimate_did_contrast_row(
                experiment_id='E3',
                provenance=provenance,
                y=y,
                cohort=cohort,
                repository=repo,
                cve=cve,
                eligibility=eligibility & reg_mask[:, None],
                observed=observed,
                observation_probability=obs_prob,
                lag=13,
                horizon_days=91,
                estimand='kev_day91_policy_regime_stratum',
                exposure='post_fix_kev_listed',
                comparator='not_yet_kev_listed_same_regime',
                subgroup=reg_name,
                multiplicity_family='E3_policy_regime_sensitivity',
                mme=mme,
                anticipation=anticipation,
                seed=study_contract.derive_child_seed(study_contract.STUDY_ID, 'E3', f'regime_{reg_name}', 0),
                fix_dyad=fix_dyad,
                calendar_shock=shock_arr,
                warning_note=note_reg,
                status_override=status_reg,
            )
            row_reg['policy_regime_verified'] = not is_unverified
            row_reg['n_exposed_cve_clusters'] = n_exp_reg
            regime_rows.append(row_reg)
    else:
        counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad, calendar_shock=shock_arr)
        regime_rows.append(_nonestimable_row(
            experiment_id='E3',
            provenance=provenance,
            estimand='kev_day91_policy_regime_stratum',
            exposure='post_fix_kev_listed',
            comparator='not_yet_kev_listed_same_regime',
            subgroup='unspecified_policy_regime',
            horizon_days=91,
            scale='risk_difference',
            uncertainty_method='two_way_cluster_influence',
            independent_unit='exposed_cve_and_repository_clusters',
            n_repositories=counts['n_repositories'],
            n_cves=counts['n_cves'],
            n_fix_dyads=counts['n_fix_dyads'],
            n_calendar_shocks=counts['n_calendar_shocks'],
            multiplicity_family='E3_policy_regime_sensitivity',
            status='NOT_IDENTIFIED',
            failure_reason='policy_regime strata absent from panel_data',
        ))

    effect_fields = list(study_contract.RESULT_ROW_COLUMNS) + [
        k for k in dict.fromkeys(key for r in effects_rows for key in r) if k not in study_contract.RESULT_ROW_COLUMNS
    ]
    regime_fields = list(study_contract.RESULT_ROW_COLUMNS) + [
        k for k in dict.fromkeys(key for r in regime_rows for key in r) if k not in study_contract.RESULT_ROW_COLUMNS
    ]
    effects_path = out_path / 'kev_effects.csv'
    counts_path = out_path / 'kev_cluster_counts.csv'
    regime_path = out_path / 'policy_regime_sensitivity.csv'
    study_contract.atomic_write_csv(effects_path, effect_fields, effects_rows)
    study_contract.atomic_write_csv(counts_path, list(cluster_count_rows[0].keys()), cluster_count_rows)
    study_contract.atomic_write_csv(regime_path, regime_fields, regime_rows)

    return {
        'rows': effects_rows,
        'primary_row': row_diff,
        'pooled_row': row_pool,
        'cluster_counts': cluster_count_rows,
        'permutation': perm_res,
        'policy_regime_rows': regime_rows,
        'artifacts': {
            'kev_effects': str(effects_path),
            'kev_cluster_counts': str(counts_path),
            'policy_regime_sensitivity': str(regime_path),
        },
    }


def _censoring_km_left_continuous(time: np.ndarray, status: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pass
    order = np.argsort(time, kind='mergesort')
    t_sorted = time[order]
    c_sorted = (status[order] == 0).astype(float)
    uniq_t = np.unique(t_sorted)
    n_total = len(t_sorted)
    g_times = []
    g_vals = []
    g_cur = 1.0
    for t_val in uniq_t:
        g_times.append(float(t_val))
        g_vals.append(float(g_cur))
        at_risk = float(np.sum(t_sorted >= t_val))
        cens_at_t = float(np.sum((t_sorted == t_val) & (c_sorted == 1.0)))
        if at_risk > 0.0 and cens_at_t > 0.0:
            g_cur *= max(0.0, 1.0 - cens_at_t / at_risk)
    return np.asarray(g_times, dtype=float), np.asarray(g_vals, dtype=float)


def _eval_left_km(t_query: float, g_times: np.ndarray, g_vals: np.ndarray) -> float:
    idx = int(np.searchsorted(g_times, t_query, side='left'))
    if idx <= 0:
        return 1.0
    if idx >= len(g_vals):
        return max(1e-8, float(g_vals[-1]))
    if abs(g_times[idx] - t_query) < 1e-12:
        return max(1e-8, float(g_vals[idx]))
    return max(1e-8, float(g_vals[idx - 1]))


def _fit_weighted_cox(
    time: np.ndarray,
    status: np.ndarray,
    design: np.ndarray,
    repository: np.ndarray,
    cve: np.ndarray,
    *,
    target_cause: int,
    horizon: float = 180.0,
    entry: np.ndarray | None = None,
    fine_gray: bool = False,
) -> dict[str, Any]:
    pass
    n, p = design.shape
    entry_arr = np.zeros(n, dtype=float) if entry is None else np.asarray(entry, dtype=float)
    events_mask = (status == target_cause) & (time <= horizon)
    n_events = int(np.sum(events_mask))
    if n_events < 2:
        return {
            'estimable': False,
            'status': 'SPARSE_ZERO_CELL',
            'failure_reason': f'Sparse/zero events (n_events={n_events}) for cause {target_cause}',
        }

                                                                      
    col0 = design[:, 0]
    if np.all(np.isin(np.unique(col0), [0.0, 1.0])) and len(np.unique(col0)) == 2:
        ev1 = int(np.sum(events_mask & (col0 == 1.0)))
        ev0 = int(np.sum(events_mask & (col0 == 0.0)))
        if ev1 == 0 or ev0 == 0:
            return {
                'estimable': False,
                'status': 'SPARSE_ZERO_CELL',
                'failure_reason': f'Zero events in one exposure arm (treated={ev1}, control={ev0}) for cause {target_cause}',
            }
    elif float(np.std(col0)) <= 1e-12:
        return {
            'estimable': False,
            'status': 'SPARSE_ZERO_CELL',
            'failure_reason': f'Zero variation in exposure covariate for cause {target_cause}',
        }

    event_times = np.sort(np.unique(time[events_mask]))
    if len(event_times) == 0:
        return {
            'estimable': False,
            'status': 'SPARSE_ZERO_CELL',
            'failure_reason': f'No event times within horizon {horizon} for cause {target_cause}',
        }

    if fine_gray:
        g_times, g_vals = _censoring_km_left_continuous(time, status)
        g_at_subject = np.array([_eval_left_km(float(time[i]), g_times, g_vals) for i in range(n)], dtype=float)
        competing_event = (status > 0) & (status != target_cause)
    else:
        g_times = np.array([], dtype=float)
        g_vals = np.array([], dtype=float)
        g_at_subject = np.ones(n, dtype=float)
        competing_event = np.zeros(n, dtype=bool)

                                                                        
    risk_weights_list: list[np.ndarray] = []
    d_m_list: list[int] = []
    fail_masks_list: list[np.ndarray] = []
    for t_val in event_times:
        f_mask = events_mask & (time == t_val)
        d_m = int(np.sum(f_mask))
        active_at_risk = (entry_arr < t_val) & (time >= t_val)
        if fine_gray:
            g_t = _eval_left_km(float(t_val), g_times, g_vals)
            sub_comp = (entry_arr < t_val) & competing_event & (time < t_val)
            w_t = np.where(active_at_risk, 1.0, np.where(sub_comp, g_t / g_at_subject, 0.0))
        else:
            w_t = active_at_risk.astype(float)
        risk_weights_list.append(w_t)
        d_m_list.append(d_m)
        fail_masks_list.append(f_mask)

    beta = np.zeros(p, dtype=float)

    def _eval_grad_hess(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        eta = np.clip(design @ b, -25.0, 25.0)
        exp_eta = np.exp(eta)
        loglik = 0.0
        grad = np.zeros(p, dtype=float)
        info = np.zeros((p, p), dtype=float)
        for t_idx in range(len(event_times)):
            w_t = risk_weights_list[t_idx]
            d_m = d_m_list[t_idx]
            f_mask = fail_masks_list[t_idx]
            wt_exp = w_t * exp_eta
            s0 = float(np.sum(wt_exp))
            if s0 <= 1e-15:
                continue
            s1 = wt_exp @ design
            x_bar = s1 / s0
            x_outer_sum = design.T @ (wt_exp[:, None] * design)
            s2 = x_outer_sum / s0
            loglik += float(np.sum(eta[f_mask])) - d_m * math.log(s0)
            grad += np.sum(design[f_mask], axis=0) - d_m * x_bar
            info += d_m * (s2 - np.outer(x_bar, x_bar))
        return loglik, grad, info

    converged = False
    for _ in range(35):
        ll_cur, grad_cur, info_cur = _eval_grad_hess(beta)
        if not np.isfinite(ll_cur) or not np.isfinite(grad_cur).all():
            break
        if float(np.linalg.norm(grad_cur, ord=np.inf)) < 1e-7:
            converged = True
            break
        if np.linalg.cond(info_cur) > 1e10:
            break
        step = np.linalg.solve(info_cur, grad_cur)
        if not np.isfinite(step).all():
            break
        step_scale = 1.0
        accepted = False
        for _ in range(15):
            cand = beta + step_scale * step
            if np.max(np.abs(cand)) > 15.0:
                step_scale *= 0.5
                continue
            ll_cand, _, _ = _eval_grad_hess(cand)
            if math.isfinite(ll_cand) and ll_cand >= ll_cur - 1e-10:
                beta = cand
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if float(np.linalg.norm(step_scale * step, ord=np.inf)) < 1e-8:
            converged = True
            break

    ll_fin, grad_fin, info_fin = _eval_grad_hess(beta)
    if not converged and float(np.linalg.norm(grad_fin, ord=np.inf)) >= 1e-5:
        return {
            'estimable': False,
            'status': 'SPARSE_ZERO_CELL',
            'failure_reason': f'Hazard model did not converge cleanly (max|grad|={np.max(np.abs(grad_fin)):.2e})',
        }
    if np.linalg.cond(info_fin) > 1e10:
        return {
            'estimable': False,
            'status': 'SPARSE_ZERO_CELL',
            'failure_reason': 'Singular information matrix in hazard fit',
        }

    inv_info = np.linalg.inv(info_fin)
    eta_hat = np.clip(design @ beta, -25.0, 25.0)
    exp_eta_hat = np.exp(eta_hat)
                                                       
    score_resid = np.zeros((n, p), dtype=float)
    for t_idx in range(len(event_times)):
        w_t = risk_weights_list[t_idx]
        d_m = d_m_list[t_idx]
        f_mask = fail_masks_list[t_idx]
        wt_exp = w_t * exp_eta_hat
        s0 = float(np.sum(wt_exp))
        if s0 <= 1e-15:
            continue
        x_bar = (wt_exp @ design) / s0
        diff_x = design - x_bar[None, :]
        score_resid[f_mask] += diff_x[f_mask]
        score_resid -= (d_m * wt_exp / s0)[:, None] * diff_x

    influence_mat = score_resid @ inv_info
    ci = _cluster_interval_from_influence(float(beta[0]), influence_mat[:, 0], repository, cve)
    return {
        'estimable': bool(ci['valid']),
        'status': ci['status'],
        'failure_reason': ci['failure_reason'],
        'beta': beta,
        'log_hr': float(beta[0]),
        'log_hr_se': ci.get('se'),
        'log_hr_lower': ci.get('lower'),
        'log_hr_upper': ci.get('upper'),
        'p_value': ci.get('p_value'),
        'df': ci.get('df'),
        'influence': influence_mat[:, 0],
        'influence_matrix': influence_mat,
        'n_events': n_events,
    }


def _coerce_competing_dict(competing_data: dict[str, Any]) -> dict[str, Any]:
    pass
    time = np.asarray(competing_data['time'], dtype=float)
    status = np.asarray(competing_data['status'])
    repo = np.asarray(competing_data['repository'])
    cve = np.asarray(competing_data['cve'])
    n = len(time)
    if not (len(status) == len(repo) == len(cve) == n) or n == 0:
        raise ValueError('Inconsistent or empty competing_data dimensions')
    entry = np.zeros(n, dtype=float) if competing_data.get('entry') is None else np.asarray(competing_data['entry'], dtype=float)
    if competing_data.get('ghsa_reviewed') is not None:
        signal = np.asarray(competing_data['ghsa_reviewed'], dtype=float)
    elif competing_data.get('treated') is not None:
        signal = np.asarray(competing_data['treated'], dtype=float)
    else:
        raise ValueError('competing_data requires "ghsa_reviewed" or "treated"')
    out = dict(competing_data)
    out.update({
        'time': time,
        'status': status,
        'repository': repo,
        'cve': cve,
        'entry': entry,
        'signal': signal,
        'horizon': float(competing_data.get('horizon', 180.0)),
    })
    return out


def run_e4_competing_risks(
    competing_data: dict[str, Any],
    out_dir: str | Path,
    *,
    config_sha256: str = '',
    input_sha256: str = '',
    run_id: str | None = None,
    mode: str = 'synthetic',
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    mme: float = 0.035,
) -> dict[str, Any]:
    pass
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    comp = _coerce_competing_dict(competing_data)
    time = comp['time']
    status = comp['status']
    repo = comp['repository']
    cve = comp['cve']
    entry = comp['entry']
    signal = comp['signal']
    horizon = comp['horizon']
    n = len(time)
    fix_dyad = np.asarray(comp['fix_dyad_id']) if comp.get('fix_dyad_id') is not None else None

    provenance = _resolve_provenance(
        'E4',
        {'bootstrap_reps': bootstrap_reps, 'mme': mme, 'horizon': horizon},
        {'n': n, 'n_repos': _unique_count(repo), 'n_cves': _unique_count(cve)},
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        run_id=run_id,
        mode=mode,
    )

    cause_labels = {
        1: 'upgrade_12',
        2: 'passive_lock_3',
        3: 'verified_removal_4',
    }
    base_counts = _cluster_counts(repo, cve, fix_dyad=fix_dyad)
    cif_rows: list[dict[str, Any]] = []
    event_curve_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []

    def _aj_diff_for_subset(
        mask: np.ndarray,
        cause_idx: int,
        estimand: str,
        exposure: str,
        comparator: str,
        subgroup: str,
        multiplicity_family: str,
    ) -> dict[str, Any]:
        cause_num = cause_idx + 1
        sub_n = int(np.sum(mask))
        sub_events = int(np.sum((status[mask] == cause_num) & (time[mask] <= horizon)))
        cnt = _cluster_counts(repo[mask], cve[mask], fix_dyad=None if fix_dyad is None else fix_dyad[mask]) if sub_n > 0 else dict(n_repositories=0, n_cves=0, n_fix_dyads=0, n_calendar_shocks=0)
        if sub_n < 4 or sub_events == 0:
            return _nonestimable_row(
                experiment_id='E4',
                provenance=provenance,
                estimand=estimand,
                exposure=exposure,
                comparator=comparator,
                subgroup=subgroup,
                horizon_days=int(horizon),
                scale='cif_difference',
                uncertainty_method='aj_product_integral_two_way_cluster',
                independent_unit='repository_x_cve',
                n_repositories=cnt['n_repositories'],
                n_cves=cnt['n_cves'],
                n_fix_dyads=cnt['n_fix_dyads'],
                n_calendar_shocks=0,
                multiplicity_family=multiplicity_family,
                status='SPARSE_ZERO_CELL',
                failure_reason=f'Zero/sparse events (n_obs={sub_n}, cause_{cause_num}_events={sub_events}) in {subgroup}',
                extra={'cause': cause_num, 'cause_label': cause_labels[cause_num], 'n_events': sub_events},
            )
        m1 = mask & (signal == 1.0)
        m0 = mask & (signal == 0.0)
        if int(np.sum(m1)) < 2 or int(np.sum(m0)) < 2:
            return _nonestimable_row(
                experiment_id='E4',
                provenance=provenance,
                estimand=estimand,
                exposure=exposure,
                comparator=comparator,
                subgroup=subgroup,
                horizon_days=int(horizon),
                scale='cif_difference',
                uncertainty_method='aj_product_integral_two_way_cluster',
                independent_unit='repository_x_cve',
                n_repositories=cnt['n_repositories'],
                n_cves=cnt['n_cves'],
                n_fix_dyads=cnt['n_fix_dyads'],
                n_calendar_shocks=0,
                multiplicity_family=multiplicity_family,
                status='SPARSE_ZERO_CELL',
                failure_reason=f'Empty signal arm in {subgroup} (n_treated={int(m1.sum())}, n_control={int(m0.sum())})',
                extra={'cause': cause_num, 'cause_label': cause_labels[cause_num], 'n_events': sub_events},
            )
        fit1 = aj_cif(time[m1], status[m1], horizon, entry=entry[m1], causes=3)
        fit0 = aj_cif(time[m0], status[m0], horizon, entry=entry[m0], causes=3)
        diff = float(fit1['cif'][cause_idx] - fit0['cif'][cause_idx])
        phi = np.zeros(n, dtype=float)
        phi[m1] = fit1['influence'][:, cause_idx]
        phi[m0] = -fit0['influence'][:, cause_idx]
        ci = _cluster_interval_from_influence(diff, phi[mask], repo[mask], cve[mask])
        mat = _materiality_diagnostics(diff if ci['valid'] else None, ci['lower'], ci['upper'], mme, ci['status'])
        return _make_result_row(
            experiment_id='E4',
            provenance=provenance,
            estimand=estimand,
            exposure=exposure,
            comparator=comparator,
            subgroup=subgroup,
            horizon_days=int(horizon),
            estimate=diff if ci['valid'] else None,
            lower=ci['lower'],
            upper=ci['upper'],
            scale='cif_difference',
            uncertainty_method='aj_product_integral_two_way_cluster',
            independent_unit='repository_x_cve',
            n_repositories=cnt['n_repositories'],
            n_cves=cnt['n_cves'],
            n_fix_dyads=cnt['n_fix_dyads'],
            n_calendar_shocks=0,
            multiplicity_family=multiplicity_family,
            test_status=ci['status'],
            failure_reason=ci['failure_reason'],
            extra={'cause': cause_num, 'cause_label': cause_labels[cause_num], 'n_events': sub_events, 'se': ci.get('se'), 'p_value': ci.get('p_value'), **mat},
        )

    def _aj_interaction_differential(
        stratum_col: str,
        estimand: str,
        exposure: str,
        comparator: str,
        subgroup: str,
        multiplicity_family: str,
        cause_idx: int = 2,
    ) -> dict[str, Any]:
        cause_num = cause_idx + 1
        if comp.get(stratum_col) is None:
            return _nonestimable_row(
                experiment_id='E4',
                provenance=provenance,
                estimand=estimand,
                exposure=exposure,
                comparator=comparator,
                subgroup=subgroup,
                horizon_days=int(horizon),
                scale='cif_difference_interaction',
                uncertainty_method='aj_product_integral_two_way_cluster',
                independent_unit='repository_x_cve',
                n_repositories=base_counts['n_repositories'],
                n_cves=base_counts['n_cves'],
                n_fix_dyads=base_counts['n_fix_dyads'],
                n_calendar_shocks=0,
                multiplicity_family=multiplicity_family,
                status='NOT_IDENTIFIED',
                failure_reason=f'Missing compatibility stratum column {stratum_col}',
                extra={'cause': cause_num, 'cause_label': cause_labels[cause_num]},
            )
        strat = np.asarray(comp[stratum_col], dtype=float)
        valid_strat = np.isfinite(strat) & np.isin(strat, [0.0, 1.0]) & np.isin(signal, [0.0, 1.0])
        total_events = int(np.sum((status[valid_strat] == cause_num) & (time[valid_strat] <= horizon)))
        if total_events == 0:
            return _nonestimable_row(
                experiment_id='E4',
                provenance=provenance,
                estimand=estimand,
                exposure=exposure,
                comparator=comparator,
                subgroup=subgroup,
                horizon_days=int(horizon),
                scale='cif_difference_interaction',
                uncertainty_method='aj_product_integral_two_way_cluster',
                independent_unit='repository_x_cve',
                n_repositories=base_counts['n_repositories'],
                n_cves=base_counts['n_cves'],
                n_fix_dyads=base_counts['n_fix_dyads'],
                n_calendar_shocks=0,
                multiplicity_family=multiplicity_family,
                status='SPARSE_ZERO_CELL',
                failure_reason=f'Zero cause_{cause_num} events across {stratum_col} strata (State 4 == 0)',
                extra={'cause': cause_num, 'cause_label': cause_labels[cause_num], 'n_events': 0},
            )
        cell_specs = ((1.0, 1.0, 1.0), (0.0, 1.0, -1.0), (1.0, 0.0, -1.0), (0.0, 0.0, 1.0))
        phi_total = np.zeros(n, dtype=float)
        est_total = 0.0
        for g_val, s_val, sign in cell_specs:
            cell_mask = valid_strat & (signal == g_val) & (strat == s_val)
            if int(np.sum(cell_mask)) < 2:
                return _nonestimable_row(
                    experiment_id='E4',
                    provenance=provenance,
                    estimand=estimand,
                    exposure=exposure,
                    comparator=comparator,
                    subgroup=subgroup,
                    horizon_days=int(horizon),
                    scale='cif_difference_interaction',
                    uncertainty_method='aj_product_integral_two_way_cluster',
                    independent_unit='repository_x_cve',
                    n_repositories=base_counts['n_repositories'],
                    n_cves=base_counts['n_cves'],
                    n_fix_dyads=base_counts['n_fix_dyads'],
                    n_calendar_shocks=0,
                    multiplicity_family=multiplicity_family,
                    status='SPARSE_ZERO_CELL',
                    failure_reason=f'Sparse/empty cell (signal={int(g_val)}, {stratum_col}={int(s_val)}, n={int(cell_mask.sum())})',
                    extra={'cause': cause_num, 'cause_label': cause_labels[cause_num], 'n_events': total_events},
                )
            fit_cell = aj_cif(time[cell_mask], status[cell_mask], horizon, entry=entry[cell_mask], causes=3)
            est_total += sign * float(fit_cell['cif'][cause_idx])
            phi_total[cell_mask] = sign * fit_cell['influence'][:, cause_idx]
        cnt = _cluster_counts(repo[valid_strat], cve[valid_strat], fix_dyad=None if fix_dyad is None else fix_dyad[valid_strat])
        ci = _cluster_interval_from_influence(est_total, phi_total[valid_strat], repo[valid_strat], cve[valid_strat])
        mat = _materiality_diagnostics(est_total if ci['valid'] else None, ci['lower'], ci['upper'], mme, ci['status'])
        return _make_result_row(
            experiment_id='E4',
            provenance=provenance,
            estimand=estimand,
            exposure=exposure,
            comparator=comparator,
            subgroup=subgroup,
            horizon_days=int(horizon),
            estimate=est_total if ci['valid'] else None,
            lower=ci['lower'],
            upper=ci['upper'],
            scale='cif_difference_interaction',
            uncertainty_method='aj_product_integral_two_way_cluster',
            independent_unit='repository_x_cve',
            n_repositories=cnt['n_repositories'],
            n_cves=cnt['n_cves'],
            n_fix_dyads=cnt['n_fix_dyads'],
            n_calendar_shocks=0,
            multiplicity_family=multiplicity_family,
            test_status=ci['status'],
            failure_reason=ci['failure_reason'],
            extra={'cause': cause_num, 'cause_label': cause_labels[cause_num], 'n_events': total_events, 'se': ci.get('se'), 'p_value': ci.get('p_value'), **mat},
        )

                                                                                                                 
    primary_row = _aj_interaction_differential(
        'out_of_range',
        'ghsa_oor_vs_inrange_removal_cif_180d_differential',
        'ghsa_x_out_of_range',
        'ghsa_x_in_range',
        'out_of_range_minus_in_range',
        'primary_confirmatory',
        cause_idx=2,
    )
    cif_rows.append(primary_row)

                                                         
    major_row = _aj_interaction_differential(
        'major_change',
        'ghsa_major_vs_nonmajor_removal_cif_180d_differential',
        'ghsa_x_major_change',
        'ghsa_x_nonmajor_change',
        'major_minus_nonmajor',
        'E4_secondary',
        cause_idx=2,
    )
    cif_rows.append(major_row)

                                                                                    
    all_mask = np.isin(signal, [0.0, 1.0])
    for c_idx in range(3):
        c_num = c_idx + 1
        cif_rows.append(_aj_diff_for_subset(
            all_mask,
            c_idx,
            f'pooled_signal_cif_180d_cause_{c_num}_{cause_labels[c_num]}',
            'ghsa_reviewed_or_treated',
            'not_yet_reviewed_or_control',
            'pooled_all',
            'E4_secondary',
        ))

                                                                                                               
    if comp.get('out_of_range') is not None:
        oor_arr = np.asarray(comp['out_of_range'], dtype=float)
        for s_val, s_lbl in ((1.0, 'out_of_range'), (0.0, 'in_range')):
            s_mask = all_mask & (oor_arr == s_val)
            for c_idx in range(3):
                c_num = c_idx + 1
                cif_rows.append(_aj_diff_for_subset(
                    s_mask,
                    c_idx,
                    f'stratum_{s_lbl}_cif_180d_cause_{c_num}_{cause_labels[c_num]}',
                    'ghsa_reviewed_or_treated',
                    'not_yet_reviewed_or_control',
                    s_lbl,
                    'E4_secondary',
                ))

    if comp.get('ecosystem') is not None:
        eco_arr = np.array([str(e).strip().lower() for e in comp['ecosystem']], dtype=object)
        for eco_name in sorted(set(eco_arr.tolist())):
            e_mask = all_mask & (eco_arr == eco_name)
            for c_idx in range(3):
                c_num = c_idx + 1
                cif_rows.append(_aj_diff_for_subset(
                    e_mask,
                    c_idx,
                    f'ecosystem_{eco_name}_cif_180d_cause_{c_num}_{cause_labels[c_num]}',
                    'ghsa_reviewed_or_treated',
                    'not_yet_reviewed_or_control',
                    eco_name,
                    'E4_ecosystem_scarcity',
                ))

                                                                             
    arm_groups: list[tuple[str, np.ndarray]] = [
        ('pooled_control', all_mask & (signal == 0.0)),
        ('pooled_treated', all_mask & (signal == 1.0)),
    ]
    if comp.get('out_of_range') is not None:
        oor_arr = np.asarray(comp['out_of_range'], dtype=float)
        for s_val, s_lbl in ((0.0, 'in_range'), (1.0, 'out_of_range')):
            arm_groups.append((f'{s_lbl}_control', all_mask & (oor_arr == s_val) & (signal == 0.0)))
            arm_groups.append((f'{s_lbl}_treated', all_mask & (oor_arr == s_val) & (signal == 1.0)))
    for grp_name, g_mask in arm_groups:
        n_g = int(np.sum(g_mask))
        c_counts = {c: int(np.sum((status[g_mask] == c) & (time[g_mask] <= horizon))) for c in (1, 2, 3)}
        n_cens = int(np.sum((status[g_mask] == 0) | (time[g_mask] > horizon)))
        if n_g >= 1:
            fit_g = aj_cif(time[g_mask], status[g_mask], horizon, entry=entry[g_mask], causes=3)
            for c_idx in range(3):
                c_num = c_idx + 1
                ev_cnt = c_counts[c_num]
                event_curve_rows.append({
                    'experiment_id': 'E4',
                    'group': grp_name,
                    'cause': c_num,
                    'cause_label': cause_labels[c_num],
                    'horizon_days': int(horizon),
                    'n_at_risk': n_g,
                    'n_events': ev_cnt,
                    'n_censored': n_cens,
                    'cif_180d': None if ev_cnt == 0 else float(fit_g['cif'][c_idx]),
                    'survival_180d': float(fit_g['survival']),
                    'test_status': 'SPARSE_ZERO_CELL' if ev_cnt == 0 else 'ESTIMABLE',
                    'failure_reason': f'Zero events for cause {c_num} in {grp_name}' if ev_cnt == 0 else '',
                })
        else:
            for c_num in (1, 2, 3):
                event_curve_rows.append({
                    'experiment_id': 'E4',
                    'group': grp_name,
                    'cause': c_num,
                    'cause_label': cause_labels[c_num],
                    'horizon_days': int(horizon),
                    'n_at_risk': 0,
                    'n_events': 0,
                    'n_censored': 0,
                    'cif_180d': None,
                    'survival_180d': None,
                    'test_status': 'SPARSE_ZERO_CELL',
                    'failure_reason': f'Empty group {grp_name}',
                })

                                                                                           
                                                                                          
    design_cols = [signal.astype(float)]
    col_names = ['signal']
    has_oor = comp.get('out_of_range') is not None
    if has_oor:
        oor_f = np.asarray(comp['out_of_range'], dtype=float)
        if float(np.std(oor_f)) > 1e-12:
            design_cols.append(oor_f)
            col_names.append('out_of_range')
    if comp.get('log1p_patch_age') is not None:
        pa = np.asarray(comp['log1p_patch_age'], dtype=float)
        if np.isfinite(pa).all() and float(np.std(pa)) > 1e-12:
            design_cols.append(pa - float(np.mean(pa)))
            col_names.append('log1p_patch_age')
    design = np.column_stack(design_cols)

    cox_fits: dict[tuple[str, int], dict[str, Any]] = {}
    for model_kind, is_fg in (('cause_specific_cox', False), ('fine_gray_ipcw_shr', True)):
        log_scale = 'subdistribution_log_hazard' if is_fg else 'log_hazard'
        hr_scale = 'subdistribution_hazard_ratio' if is_fg else 'hazard_ratio'
        for c_num in (1, 2, 3):
            fit_h = _fit_weighted_cox(
                time,
                status,
                design,
                repo,
                cve,
                target_cause=c_num,
                horizon=horizon,
                entry=entry,
                fine_gray=is_fg,
            )
            cox_fits[(model_kind, c_num)] = fit_h
            if fit_h['estimable']:
                log_est = float(fit_h['log_hr'])
                log_lo = float(fit_h['log_hr_lower'])
                log_hi = float(fit_h['log_hr_upper'])
                hr_est = float(math.exp(log_est))
                hr_lo = float(math.exp(log_lo))
                hr_hi = float(math.exp(log_hi))
                                      
                hazard_rows.append(_make_result_row(
                    experiment_id='E4',
                    provenance=provenance,
                    estimand=f'{model_kind}_log_hr_cause_{c_num}_{cause_labels[c_num]}',
                    exposure='ghsa_reviewed_or_treated',
                    comparator='not_yet_reviewed_or_control',
                    subgroup=cause_labels[c_num],
                    horizon_days=int(horizon),
                    estimate=log_est,
                    lower=log_lo,
                    upper=log_hi,
                    scale=log_scale,
                    uncertainty_method='lin_wei_two_way_cluster_sandwich',
                    independent_unit='repository_x_cve',
                    n_repositories=base_counts['n_repositories'],
                    n_cves=base_counts['n_cves'],
                    n_fix_dyads=base_counts['n_fix_dyads'],
                    n_calendar_shocks=0,
                    multiplicity_family='E4_secondary_hazards',
                    test_status=fit_h['status'],
                    failure_reason=fit_h['failure_reason'],
                    extra={'cause': c_num, 'n_events': fit_h['n_events'], 'se': fit_h['log_hr_se'], 'p_value': fit_h['p_value']},
                ))
                                                                   
                hazard_rows.append(_make_result_row(
                    experiment_id='E4',
                    provenance=provenance,
                    estimand=f'{model_kind}_hr_cause_{c_num}_{cause_labels[c_num]}',
                    exposure='ghsa_reviewed_or_treated',
                    comparator='not_yet_reviewed_or_control',
                    subgroup=cause_labels[c_num],
                    horizon_days=int(horizon),
                    estimate=hr_est,
                    lower=hr_lo,
                    upper=hr_hi,
                    scale=hr_scale,
                    uncertainty_method='lin_wei_two_way_cluster_sandwich_exp_scale_consistent',
                    independent_unit='repository_x_cve',
                    n_repositories=base_counts['n_repositories'],
                    n_cves=base_counts['n_cves'],
                    n_fix_dyads=base_counts['n_fix_dyads'],
                    n_calendar_shocks=0,
                    multiplicity_family='E4_secondary_hazards',
                    test_status=fit_h['status'],
                    failure_reason='FC-06_scale_consistent_exp_beta_interval_not_causal_cif',
                    extra={'cause': c_num, 'n_events': fit_h['n_events'], 'log_hr_se': fit_h['log_hr_se'], 'p_value': fit_h['p_value']},
                ))
            else:
                for sc, tag in ((log_scale, 'log_hr'), (hr_scale, 'hr')):
                    hazard_rows.append(_nonestimable_row(
                        experiment_id='E4',
                        provenance=provenance,
                        estimand=f'{model_kind}_{tag}_cause_{c_num}_{cause_labels[c_num]}',
                        exposure='ghsa_reviewed_or_treated',
                        comparator='not_yet_reviewed_or_control',
                        subgroup=cause_labels[c_num],
                        horizon_days=int(horizon),
                        scale=sc,
                        uncertainty_method='lin_wei_two_way_cluster_sandwich',
                        independent_unit='repository_x_cve',
                        n_repositories=base_counts['n_repositories'],
                        n_cves=base_counts['n_cves'],
                        n_fix_dyads=base_counts['n_fix_dyads'],
                        n_calendar_shocks=0,
                        multiplicity_family='E4_secondary_hazards',
                        status=fit_h['status'],
                        failure_reason=fit_h['failure_reason'],
                        extra={'cause': c_num, 'n_events': int(np.sum((status == c_num) & (time <= horizon)))},
                    ))

                                                                                               
        fit_num = cox_fits.get((model_kind, 3), {})
        fit_den = cox_fits.get((model_kind, 1), {})
        ratio_est_id = f'{model_kind}_log_coef_ratio_removal_to_upgrade'
        if fit_num.get('estimable') and fit_den.get('estimable'):
            b_a = float(fit_num['log_hr'])
            b_b = float(fit_den['log_hr'])
            if abs(b_b) < 1e-4:
                hazard_rows.append(_nonestimable_row(
                    experiment_id='E4',
                    provenance=provenance,
                    estimand=ratio_est_id,
                    exposure='log_coef_cause_3_over_cause_1',
                    comparator='delta_method_joint_cluster_covariance',
                    subgroup='verified_removal_over_upgrade',
                    horizon_days=int(horizon),
                    scale='log_hazard_ratio_not_saved_effort',
                    uncertainty_method='delta_method_two_way_cluster_covariance',
                    independent_unit='repository_x_cve',
                    n_repositories=base_counts['n_repositories'],
                    n_cves=base_counts['n_cves'],
                    n_fix_dyads=base_counts['n_fix_dyads'],
                    n_calendar_shocks=0,
                    multiplicity_family='E4_secondary_hazards',
                    status='NOT_ESTIMABLE',
                    failure_reason=f'FC-07 near-zero denominator log-HR ({b_b:.3e}) prevents finite delta ratio',
                ))
            else:
                cov_mat = _cluster_covariance_matrix(
                    [np.asarray(fit_num['influence'], dtype=float), np.asarray(fit_den['influence'], dtype=float)],
                    repo,
                    cve,
                )
                grad_g = np.array([1.0 / b_b, -b_a / (b_b * b_b)], dtype=float)
                var_ratio = float(grad_g @ cov_mat @ grad_g)
                if not math.isfinite(var_ratio) or var_ratio <= 0.0:
                    hazard_rows.append(_nonestimable_row(
                        experiment_id='E4',
                        provenance=provenance,
                        estimand=ratio_est_id,
                        exposure='log_coef_cause_3_over_cause_1',
                        comparator='delta_method_joint_cluster_covariance',
                        subgroup='verified_removal_over_upgrade',
                        horizon_days=int(horizon),
                        scale='log_hazard_ratio_not_saved_effort',
                        uncertainty_method='delta_method_two_way_cluster_covariance',
                        independent_unit='repository_x_cve',
                        n_repositories=base_counts['n_repositories'],
                        n_cves=base_counts['n_cves'],
                        n_fix_dyads=base_counts['n_fix_dyads'],
                        n_calendar_shocks=0,
                        multiplicity_family='E4_secondary_hazards',
                        status='NOT_ESTIMABLE',
                        failure_reason=f'Nonpositive delta-method variance ({var_ratio}) for log-coefficient ratio',
                    ))
                else:
                    ratio_val = float(b_a / b_b)
                    se_ratio = float(math.sqrt(var_ratio))
                    df_r = max(1, min(base_counts['n_repositories'], base_counts['n_cves']) - 1)
                    crit_r = float(student_t.ppf(0.975, df_r))
                    hazard_rows.append(_make_result_row(
                        experiment_id='E4',
                        provenance=provenance,
                        estimand=ratio_est_id,
                        exposure='log_coef_cause_3_over_cause_1',
                        comparator='delta_method_joint_cluster_covariance',
                        subgroup='verified_removal_over_upgrade',
                        horizon_days=int(horizon),
                        estimate=ratio_val,
                        lower=float(ratio_val - crit_r * se_ratio),
                        upper=float(ratio_val + crit_r * se_ratio),
                        scale='log_hazard_ratio_not_saved_effort',
                        uncertainty_method='delta_method_two_way_cluster_covariance',
                        independent_unit='repository_x_cve',
                        n_repositories=base_counts['n_repositories'],
                        n_cves=base_counts['n_cves'],
                        n_fix_dyads=base_counts['n_fix_dyads'],
                        n_calendar_shocks=0,
                        multiplicity_family='E4_secondary_hazards',
                        test_status='ESTIMABLE',
                        failure_reason='FC-07_log_coefficient_ratio_only_not_saved_effort',
                        extra={
                            'se': se_ratio,
                            'cov_num_den': float(cov_mat[0, 1]),
                            'var_num': float(cov_mat[0, 0]),
                            'var_den': float(cov_mat[1, 1]),
                        },
                    ))
        else:
            hazard_rows.append(_nonestimable_row(
                experiment_id='E4',
                provenance=provenance,
                estimand=ratio_est_id,
                exposure='log_coef_cause_3_over_cause_1',
                comparator='delta_method_joint_cluster_covariance',
                subgroup='verified_removal_over_upgrade',
                horizon_days=int(horizon),
                scale='log_hazard_ratio_not_saved_effort',
                uncertainty_method='delta_method_two_way_cluster_covariance',
                independent_unit='repository_x_cve',
                n_repositories=base_counts['n_repositories'],
                n_cves=base_counts['n_cves'],
                n_fix_dyads=base_counts['n_fix_dyads'],
                n_calendar_shocks=0,
                multiplicity_family='E4_secondary_hazards',
                status='SPARSE_ZERO_CELL',
                failure_reason='Cannot form FC-07 log-coefficient ratio because numerator or denominator hazard fit is nonestimable',
            ))

    cif_fields = list(study_contract.RESULT_ROW_COLUMNS) + [
        k for k in dict.fromkeys(key for r in cif_rows for key in r) if k not in study_contract.RESULT_ROW_COLUMNS
    ]
    hazard_fields = list(study_contract.RESULT_ROW_COLUMNS) + [
        k for k in dict.fromkeys(key for r in hazard_rows for key in r) if k not in study_contract.RESULT_ROW_COLUMNS
    ]
    cif_path = out_path / 'cif_180d.csv'
    events_path = out_path / 'competing_events.csv'
    hazard_path = out_path / 'hazard_secondary.csv'
    study_contract.atomic_write_csv(cif_path, cif_fields, cif_rows)
    study_contract.atomic_write_csv(events_path, list(event_curve_rows[0].keys()), event_curve_rows)
    study_contract.atomic_write_csv(hazard_path, hazard_fields, hazard_rows)

    return {
        'rows': cif_rows,
        'primary_row': primary_row,
        'major_row': major_row,
        'competing_events': event_curve_rows,
        'hazard_rows': hazard_rows,
        'artifacts': {
            'cif_180d': str(cif_path),
            'competing_events': str(events_path),
            'hazard_secondary': str(hazard_path),
        },
    }


def _build_synthetic_did_panel(
    *,
    seed: int,
    nrepo: int,
    ncve: int,
    nt: int,
    cohorts: tuple[float, ...],
    group_effects: dict[int, float],
    reentry_persistence: float = 0.0,
    unknown_rate: float = 0.02,
    extra_bot_interaction: float = 0.0,
    ecosystems: tuple[str, ...] = ('maven', 'npm', 'pypi'),
    n_exposed_cves_cap: int | None = None,
) -> dict[str, Any]:
    pass
    rng = np.random.default_rng(int(seed))
    repo, cve = synthetic_cohorts.cluster_grid(nrepo, ncve)
    n = len(repo)

                                                          
    cve_eco = np.array([ecosystems[c % len(ecosystems)] for c in range(ncve)], dtype=object)
    allocation = np.full(ncve, np.inf, dtype=float)
    if n_exposed_cves_cap is not None:
                                                                                        
        n_exp = min(ncve - len(ecosystems), int(n_exposed_cves_cap))
        chosen_list: list[int] = []
        per_eco = max(2, n_exp // len(ecosystems))
        for eco_idx, eco_name in enumerate(ecosystems):
            eco_cves = np.flatnonzero(cve_eco == eco_name)
            take = min(len(eco_cves) - 2, per_eco if eco_idx < len(ecosystems) - 1 else (n_exp - len(chosen_list)))
            if take > 0:
                chosen_list.extend(rng.choice(eco_cves, size=take, replace=False).tolist())
        for idx, c_id in enumerate(chosen_list[:n_exp]):
            allocation[int(c_id)] = cohorts[idx % len(cohorts)]
    else:
        for c_id in range(ncve):
            slot = c_id % (len(cohorts) + 1)
            allocation[c_id] = cohorts[slot] if slot < len(cohorts) else np.inf
        rng.shuffle(allocation)
    cohort = allocation[cve]
    ecosystem = cve_eco[cve]

    x = rng.integers(0, 2, size=n)
    cvss_score = np.clip(4.0 + 2.5 * x + rng.normal(1.2, 0.8, size=n), 2.0, 9.9)
    calendar_period = (cve % 3).astype(int)
    matched_block_id = np.array([f'blk_{int(cvss_score[i] >= 7.0)}' for i in range(n)], dtype=object)

                                                                                                
    bot_prob = 0.35 + 0.25 * x
    pre_signal_bot = (rng.random(n) < bot_prob).astype(float)

    cve_shock = rng.uniform(-0.04, 0.04, size=(ncve, nt))
    repo_shock = rng.uniform(-0.05, 0.05, size=(nrepo, nt))
    base = 0.24 + 0.04 * x[:, None] + 0.001 * np.arange(nt)[None, :]
    untreated = base + cve_shock[cve, :] + repo_shock[repo, :]

    effects = np.zeros((n, nt), dtype=float)
    for g_val, tau in group_effects.items():
        g_int = int(g_val)
        sel = cohort == float(g_int)
        for t in range(nt):
            lag = t - g_int
            if lag >= 0:
                ramp = min(1.0, (lag + 1) / 7.0)
                effects[sel, t] = (tau + extra_bot_interaction * pre_signal_bot[sel]) * ramp

    probability = untreated + effects
    if np.any(probability <= 0.0) or np.any(probability >= 1.0):
        raise ArithmeticError('Synthetic study DGP probability leaves (0, 1); no clipping allowed')

    if reentry_persistence > 0.0:
        y = np.zeros((n, nt), dtype=float)
        y[:, 0] = rng.binomial(1, probability[:, 0])
        for t in range(1, nt):
            p_t = reentry_persistence * y[:, t - 1] + (1.0 - reentry_persistence) * probability[:, t]
            if np.any(p_t <= 0.0) or np.any(p_t >= 1.0):
                raise ArithmeticError('Reentry transition probability leaves (0, 1)')
            y[:, t] = rng.binomial(1, p_t)
    else:
        y = rng.binomial(1, probability).astype(float)

                                                                                                                      
    y_day90 = y.copy()
    flip_90 = (y == 1.0) & (rng.random((n, nt)) < 0.05)
    y_day90[flip_90] = 0.0
    y_upgrades_only = np.where((y == 1.0) & (rng.random((n, nt)) < 0.82), 1.0, 0.0)
    y_first_response = np.maximum.accumulate(y, axis=1)

    eligibility = np.ones((n, nt), dtype=bool)
    observed = rng.random((n, nt)) >= float(unknown_rate)
                                                                                                     
    y_out = np.where(observed, y, np.nan)
    y90_out = np.where(observed, y_day90, np.nan)
    y_upg_out = np.where(observed, y_upgrades_only, np.nan)
    y_first_out = np.where(observed, y_first_response, np.nan)

    pre_poc_present = rng.random(n) < 0.08
    organic_cohort = np.where(np.isfinite(cohort), np.clip(cohort + rng.integers(-1, 2, size=n), 2, nt - 14), np.inf)
    poc_cohort = np.where((cve % 5 == 0), float(cohorts[1 if len(cohorts) > 1 else 0]), np.inf)

    policy_choices = np.array(['bod_22_01_verified', 'bod_26_04_transition_unverified', 'pre_directive'], dtype=object)
    policy_regime = policy_choices[(cve // len(ecosystems)) % 3]
    policy_regime_verified = policy_regime != 'bod_26_04_transition_unverified'
    obs_keys = [(f'repo_{int(repo[i])}', 'proj_main', f'dep_{int(i % 5)}', f'cve_{int(cve[i])}', f'anc_{i}') for i in range(n)]
    fix_dyad = np.array([f'{int(repo[i])}:{int(cve[i])}' for i in range(n)], dtype=object)

    return {
        'y': y_out,
        'y_day90': y90_out,
        'y_upgrades_only': y_upg_out,
        'y_first_response': y_first_out,
        'cohort': cohort,
        'organic_cohort': organic_cohort,
        'poc_cohort': poc_cohort,
        'pre_poc_present': pre_poc_present,
        'repository': repo,
        'cve': cve,
        'fix_dyad_id': fix_dyad,
        'observation_keys': obs_keys,
        'ecosystem': ecosystem,
        'matched_block_id': matched_block_id,
        'cvss_score': cvss_score,
        'calendar_period': calendar_period,
        'pre_signal_bot': pre_signal_bot,
        'bot_propensity': bot_prob,
        'policy_regime': policy_regime,
        'policy_regime_verified': policy_regime_verified,
        'eligibility': eligibility,
        'observed': observed,
        'observation_probability': np.ones((n, nt), dtype=float),
    }


def _build_synthetic_competing_data(
    *,
    seed: int,
    nrepo: int = 32,
    ncve: int = 120,
    zero_state4: bool = False,
    maven_scarcity: bool = False,
) -> dict[str, Any]:
    pass
    rng = np.random.default_rng(int(seed))
    repo, cve = synthetic_cohorts.cluster_grid(nrepo, ncve)
    n = len(repo)
    exposed_cve = np.repeat([0, 1], ncve // 2)
    if len(exposed_cve) < ncve:
        exposed_cve = np.concatenate([exposed_cve, np.zeros(ncve - len(exposed_cve), dtype=int)])
    rng.shuffle(exposed_cve)
    ghsa_reviewed = exposed_cve[cve].astype(float)
    out_of_range = rng.integers(0, 2, size=n).astype(float)
    major_change = np.where(rng.random(n) < 0.7, out_of_range, 1.0 - out_of_range).astype(float)
    eco_pool = np.array(['maven', 'npm', 'pypi'], dtype=object)
    ecosystem = eco_pool[cve % 3]
    log1p_patch_age = np.log1p(rng.uniform(1.0, 90.0, size=n))

    zc = rng.choice([-1.0, 1.0], size=ncve)
    zr = rng.choice([-1.0, 1.0], size=nrepo)
    frailty = 0.004 * zc[cve] + 0.003 * zr[repo]

                                                                                               
    h1 = 0.065 + 0.030 * ghsa_reviewed * (1.0 - 0.5 * out_of_range) + frailty
    h2 = 0.035 + 0.008 * ghsa_reviewed + 0.5 * frailty
    h3 = 0.018 + 0.014 * ghsa_reviewed + 0.030 * ghsa_reviewed * out_of_range + 0.006 * out_of_range + frailty

    if zero_state4:
        h3 = np.zeros(n, dtype=float)
    if maven_scarcity:
        is_mav = ecosystem == 'maven'
        h2 = np.where(is_mav, 0.0, h2)
        h3 = np.where(is_mav, 0.0, h3)

    hazards = np.column_stack([h1, h2, h3])
    if np.any(hazards < 0.0) or np.any(hazards.sum(axis=1) >= 1.0):
        raise ArithmeticError('Competing-risk hazard sum leaves [0, 1); no clipping allowed')

    time = np.full(n, 180.0, dtype=float)
    status = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for day in (30.0, 60.0, 90.0, 120.0, 150.0, 180.0):
        draw = rng.random(n)
        cutoffs = hazards.cumsum(axis=1)
        for cause in (1, 2, 3):
            low = cutoffs[:, cause - 2] if cause > 1 else 0.0
            hit = alive & (draw < cutoffs[:, cause - 1]) & (draw >= low)
            time[hit] = day
            status[hit] = cause
        alive &= draw >= cutoffs[:, -1]

    censor = rng.exponential(700.0, size=n)
    lost = censor < time
    time[lost] = np.maximum(1.0, censor[lost])
    status[lost] = 0

    return {
        'time': time,
        'status': status,
        'treated': ghsa_reviewed,
        'ghsa_reviewed': ghsa_reviewed,
        'out_of_range': out_of_range,
        'major_change': major_change,
        'ecosystem': ecosystem,
        'log1p_patch_age': log1p_patch_age,
        'repository': repo,
        'cve': cve,
        'fix_dyad_id': np.array([f'{int(repo[i])}:{int(cve[i])}' for i in range(n)], dtype=object),
        'horizon': 180.0,
    }


def generate_synthetic_study_signals(seed: int = study_contract.MASTER_SEED) -> dict[str, Any]:
    pass
    s_b0 = int(study_contract.derive_child_seed(study_contract.STUDY_ID, 'SYN', 'B0_E1', int(seed) % 10000))
    s_e2 = int(study_contract.derive_child_seed(study_contract.STUDY_ID, 'SYN', 'E2', int(seed) % 10000))
    s_e3 = int(study_contract.derive_child_seed(study_contract.STUDY_ID, 'SYN', 'E3', int(seed) % 10000))
    s_e3_few = int(study_contract.derive_child_seed(study_contract.STUDY_ID, 'SYN', 'E3_few', int(seed) % 10000))
    s_e4 = int(study_contract.derive_child_seed(study_contract.STUDY_ID, 'SYN', 'E4', int(seed) % 10000))
    s_e4_sp = int(study_contract.derive_child_seed(study_contract.STUDY_ID, 'SYN', 'E4_sparse', int(seed) % 10000))

                                                                                                                          
    e1_panel = _build_synthetic_did_panel(
        seed=s_b0,
        nrepo=32,
        ncve=120,
        nt=27,
        cohorts=(4.0, 8.0, 12.0),
        group_effects={4: 0.012, 8: 0.015, 12: 0.018},
        reentry_persistence=0.55,
        unknown_rate=0.02,
    )

                                                                                                 
    e2_panel = _build_synthetic_did_panel(
        seed=s_e2,
        nrepo=32,
        ncve=120,
        nt=36,
        cohorts=(14.0, 18.0, 22.0),
        group_effects={14: 0.025, 18: 0.030, 22: 0.035},
        extra_bot_interaction=0.055,
        reentry_persistence=0.45,
        unknown_rate=0.01,
    )

                                                                                             
    e3_panel = _build_synthetic_did_panel(
        seed=s_e3,
        nrepo=36,
        ncve=144,
        nt=27,
        cohorts=(4.0, 8.0, 12.0),
        group_effects={4: 0.04, 8: 0.05, 12: 0.06},
        unknown_rate=0.01,
    )
    e3_few_cluster_panel = _build_synthetic_did_panel(
        seed=s_e3_few,
        nrepo=24,
        ncve=36,
        nt=27,
        cohorts=(4.0, 8.0, 12.0),
        group_effects={4: 0.04, 8: 0.05, 12: 0.06},
        unknown_rate=0.01,
        n_exposed_cves_cap=16,
    )

                                                                                                    
    e4_competing = _build_synthetic_competing_data(seed=s_e4, nrepo=32, ncve=120)
    e4_sparse_state4 = _build_synthetic_competing_data(seed=s_e4_sp, nrepo=20, ncve=60, zero_state4=True)
    e4_sparse_maven = _build_synthetic_competing_data(seed=s_e4_sp + 1, nrepo=20, ncve=60, maven_scarcity=True)

    return {
        'b0_panel': e1_panel,
        'e1_panel': e1_panel,
        'e2_panel': e2_panel,
        'e3_panel': e3_panel,
        'e3_few_cluster_panel': e3_few_cluster_panel,
        'e4_competing': e4_competing,
        'e4_sparse_state4': e4_sparse_state4,
        'e4_sparse_maven': e4_sparse_maven,
    }


def run_all_signal_models(
    out_dir: str | Path,
    *,
    mode: str = 'synthetic',
    seed: int = study_contract.MASTER_SEED,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    perm_reps: int = DEFAULT_PERM_REPS,
    study_inputs: dict[str, Any] | None = None,
    input_paths: tuple[Path | str, ...] = (),
    config_sha256: str = '',
    input_sha256: str = '',
) -> dict[str, Any]:
    pass
    if mode not in ('synthetic', 'official'):
        raise ValueError(f'Unsupported execution mode {mode!r}; expected synthetic or official')
    if mode == 'official':
        validation.assert_official_empirical_unlocked()
        if study_inputs is None:
            raise ValueError('Official empirical execution requires explicit study_inputs')
    else:
        if input_paths:
            validation.assert_synthetic_mode_isolated(input_paths)
        if study_inputs is None:
            study_inputs = generate_synthetic_study_signals(seed=seed)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    b0_res = run_b0_baselines(
        study_inputs['b0_panel'],
        out_path,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        mode=mode,
        bootstrap_reps=bootstrap_reps,
    )
    e1_res = run_e1_epss(
        study_inputs['e1_panel'],
        out_path,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        mode=mode,
        bootstrap_reps=bootstrap_reps,
    )
    e2_res = run_e2_ghsa_bot(
        study_inputs['e2_panel'],
        out_path,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        mode=mode,
        bootstrap_reps=bootstrap_reps,
    )
    e3_res = run_e3_kev_ecosystem(
        study_inputs['e3_panel'],
        out_path,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        mode=mode,
        perm_reps=perm_reps,
        bootstrap_reps=bootstrap_reps,
    )
    e4_res = run_e4_competing_risks(
        study_inputs['e4_competing'],
        out_path,
        config_sha256=config_sha256,
        input_sha256=input_sha256,
        mode=mode,
        bootstrap_reps=bootstrap_reps,
    )

    return {
        'mode': mode,
        'B0': b0_res,
        'E1': e1_res,
        'E2': e2_res,
        'E3': e3_res,
        'E4': e4_res,
        'artifacts': {
            **b0_res['artifacts'],
            **e1_res['artifacts'],
            **e2_res['artifacts'],
            **e3_res['artifacts'],
            **e4_res['artifacts'],
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Execute B0, E1, E2, E3, and E4 signal estimation models.',
    )
    parser.add_argument('-m', choices=['synthetic', 'official'], default='synthetic', help='Execution mode')
    parser.add_argument('-o', required=True, help='Output directory for CSV and JSON artifacts')
    parser.add_argument('-s', type=int, default=study_contract.MASTER_SEED, help='Master random seed')
    parser.add_argument('-b', type=int, default=DEFAULT_BOOTSTRAP_REPS, help='Bootstrap replicate count')
    parser.add_argument('-p', type=int, default=DEFAULT_PERM_REPS, help='Permutation replicate count')
    parser.add_argument('-c', default='', help='Optional frozen config path for SHA256 provenance')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg_sha = study_contract.sha256_file(Path(args.c)) if args.c else ''
    result = run_all_signal_models(
        args.o,
        mode=args.m,
        seed=args.s,
        bootstrap_reps=args.b,
        perm_reps=args.p,
        config_sha256=cfg_sha,
    )
    print(json.dumps({
        'status': 'COMPLETED',
        'mode': result['mode'],
        'artifact_count': len(result['artifacts']),
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
