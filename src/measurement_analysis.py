"""Recompute measurement contrasts from the released empirical inputs."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, t as student_t


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = Path(__file__).resolve().parent.parent
OUT_85 = Path(__file__).resolve().parent.parent / "data"
I0_DIR = Path(__file__).resolve().parent
PIPE_DIR = Path(__file__).resolve().parent
COMPANION_DIR = Path(__file__).resolve().parent.parent / "results" / "recomputed" / "measurement"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


estimators = _load_module("_comp_i0_estimators", I0_DIR / "estimators.py")
study_contract = _load_module("study_contract", PIPE_DIR / "study_contract.py")
validation = _load_module("validation", PIPE_DIR / "validation.py")
signal_models = _load_module("_comp_signal_models", PIPE_DIR / "signal_models.py")
version_rules = _load_module("_comp_version_rules", Path(__file__).resolve().parent / "version_rules.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_owner(repo_url: str) -> str:
    s = str(repo_url or "").strip().rstrip("/")
    parts = s.split("/")
    if len(parts) >= 2:
        return parts[-2].lower()
    return s.lower()


def strip_tid_cve(cve_raw: str) -> str:
    return str(cve_raw or "").split(":")[0].strip()


def two_way_mean_ci(
    values: np.ndarray,
    cluster_a: np.ndarray,
    cluster_b: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return {"n": int(n), "estimate": None, "se": None, "lower": None, "upper": None, "p_value": None}
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    sel = np.isfinite(v) & (w > 0)
    if int(sel.sum()) < 2:
        return {"n": int(sel.sum()), "estimate": None, "se": None, "lower": None, "upper": None, "p_value": None}
    est, inf = estimators.weighted_mean_influence(v, sel, w)
    ca = np.asarray(cluster_a)[sel]
    cb = np.asarray(cluster_b)[sel]
    inf_s = inf[sel]
    n_a = int(len(np.unique(ca)))
    n_b = int(len(np.unique(cb)))
    if n_a < 2 or n_b < 2:
        return {
            "n": int(sel.sum()),
            "n_cluster_a": n_a,
            "n_cluster_b": n_b,
            "estimate": float(est),
            "se": None,
            "lower": None,
            "upper": None,
            "p_value": None,
        }
    var = estimators.cluster_variance(inf_s, ca, cb, adjust=True)
    if not math.isfinite(var) or var <= 0.0:
        return {
            "n": int(sel.sum()),
            "n_cluster_a": n_a,
            "n_cluster_b": n_b,
            "estimate": float(est),
            "se": None,
            "lower": None,
            "upper": None,
            "p_value": None,
        }
    df = max(1, min(n_a, n_b) - 1)
    se = math.sqrt(var)
    crit = float(student_t.ppf(0.975, df))
    p_val = float(2.0 * student_t.sf(abs(est / se), df))
    return {
        "n": int(sel.sum()),
        "n_cluster_a": n_a,
        "n_cluster_b": n_b,
        "estimate": float(est),
        "se": float(se),
        "lower": float(est - crit * se),
        "upper": float(est + crit * se),
        "p_value": p_val,
    }


def diff_in_means_ci(
    values: np.ndarray,
    treated_mask: np.ndarray,
    control_mask: np.ndarray,
    cluster_a: np.ndarray,
    cluster_b: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    n = len(v)
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    t_sel = np.asarray(treated_mask, dtype=bool) & np.isfinite(v) & (w > 0)
    c_sel = np.asarray(control_mask, dtype=bool) & np.isfinite(v) & (w > 0)
    if int(t_sel.sum()) < 2 or int(c_sel.sum()) < 2:
        return {"estimable": False, "n_treated": int(t_sel.sum()), "n_control": int(c_sel.sum())}
    mt, it = estimators.weighted_mean_influence(v, t_sel, w)
    mc, ic = estimators.weighted_mean_influence(v, c_sel, w)
    est = float(mt - mc)
    used = t_sel | c_sel
    phi = (it - ic)[used]
    ca = np.asarray(cluster_a)[used]
    cb = np.asarray(cluster_b)[used]
    n_a = int(len(np.unique(ca)))
    n_b = int(len(len(np.unique(cb)) and np.unique(cb)))
    var = estimators.cluster_variance(phi, ca, cb, adjust=True)
    if not math.isfinite(var) or var <= 0.0:
        return {
            "estimable": False,
            "estimate": est,
            "treated_mean": float(mt),
            "control_mean": float(mc),
            "n_treated": int(t_sel.sum()),
            "n_control": int(c_sel.sum()),
        }
    df = max(1, min(n_a, n_b) - 1)
    se = math.sqrt(var)
    crit = float(student_t.ppf(0.975, df))
    p_val = float(2.0 * student_t.sf(abs(est / se), df))
    return {
        "estimable": True,
        "estimate": est,
        "treated_mean": float(mt),
        "control_mean": float(mc),
        "se": float(se),
        "lower": float(est - crit * se),
        "upper": float(est + crit * se),
        "p_value": p_val,
        "n_treated": int(t_sel.sum()),
        "n_control": int(c_sel.sum()),
        "n_cluster_a": n_a,
        "n_cluster_b": n_b,
    }


def manski_contrast_bounds(
    y_obs: np.ndarray,
    is_obs: np.ndarray,
    treated_mask: np.ndarray,
    control_mask: np.ndarray,
) -> dict[str, Any]:
    t = np.asarray(treated_mask, dtype=bool)
    c = np.asarray(control_mask, dtype=bool)
    r = np.asarray(is_obs, dtype=bool)
    y = np.asarray(y_obs, dtype=float)

    n_t = int(t.sum())
    n_t_obs = int((t & r).sum())
    p_t_obs = n_t_obs / n_t if n_t else 0.0
    mu_t_obs = float(np.mean(y[t & r])) if n_t_obs else 0.0
    t_low = p_t_obs * mu_t_obs + (1.0 - p_t_obs) * 0.0
    t_high = p_t_obs * mu_t_obs + (1.0 - p_t_obs) * 1.0

    n_c = int(c.sum())
    n_c_obs = int((c & r).sum())
    p_c_obs = n_c_obs / n_c if n_c else 0.0
    mu_c_obs = float(np.mean(y[c & r])) if n_c_obs else 0.0
    c_low = p_c_obs * mu_c_obs + (1.0 - p_c_obs) * 0.0
    c_high = p_c_obs * mu_c_obs + (1.0 - p_c_obs) * 1.0

    return {
        "n_treated_total": n_t,
        "n_treated_observed": n_t_obs,
        "treated_recoverability": float(p_t_obs),
        "treated_complete_case_mean": float(mu_t_obs),
        "n_control_total": n_c,
        "n_control_observed": n_c_obs,
        "control_recoverability": float(p_c_obs),
        "control_complete_case_mean": float(mu_c_obs),
        "recoverability_gap_treated_minus_control": float(p_t_obs - p_c_obs),
        "complete_case_diff": float(mu_t_obs - mu_c_obs),
        "manski_lower": float(t_low - c_high),
        "manski_upper": float(t_high - c_low),
    }


def fit_logistic_ipw(X: np.ndarray, r: np.ndarray, l2: float = 1.0) -> np.ndarray:
    y = np.asarray(r, dtype=float)
    n, p = X.shape

    def loss_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(X @ beta, -25.0, 25.0)
        prob = expit(eta)
        nll = -float(np.sum(y * np.log(np.clip(prob, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - prob, 1e-12, 1.0))))
        reg = 0.5 * l2 * float(np.sum(beta[1:] ** 2))
        grad = X.T @ (prob - y)
        grad[1:] += l2 * beta[1:]
        return nll + reg, grad

    res = minimize(loss_and_grad, np.zeros(p, dtype=float), jac=True, method="BFGS", options={"maxiter": 300})
    probs = expit(np.clip(X @ res.x, -25.0, 25.0))
    return np.clip(probs, 0.03, 0.97)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    COMPANION_DIR.mkdir(parents=True, exist_ok=True)
    git_obs = pd.read_parquet(OUT_85 / "git_observations.parquet")
    first_exit = pd.read_parquet(OUT_85 / "first_exit.parquet")
    covs = pd.read_parquet(OUT_85 / "covariates.parquet")
    with (OUT_85 / "measurement_variants.json").open(encoding="utf-8") as f:
        a0_list = json.load(f)
    a0_df = pd.DataFrame(a0_list)
    with (OUT_85 / "robustness_panel.json").open(encoding="utf-8") as f:
        r0_df = pd.DataFrame(json.load(f))

    e1_npz = dict(np.load(OUT_85 / "signal_panels" / "epss_panel.npz", allow_pickle=False))
    e2_npz = dict(np.load(OUT_85 / "signal_panels" / "advisory_panel.npz", allow_pickle=False))
    e3_npz = dict(np.load(OUT_85 / "signal_panels" / "kev_panel.npz", allow_pickle=False))
    e4_npz = dict(np.load(OUT_85 / "signal_panels" / "competing_events.npz", allow_pickle=False))

    n_total = len(git_obs)
    assert len(first_exit) == n_total == len(covs) == len(a0_df) == len(r0_df)

    repo_arr = git_obs["repository_id"].astype(str).to_numpy()
    owner_arr = np.array([extract_owner(u) for u in repo_arr], dtype=object)
    true_cve_arr = git_obs["cve_id"].astype(str).to_numpy()
    eco_arr = git_obs["ecosystem"].astype(str).str.lower().to_numpy()
    obs_status_arr = git_obs["observation_status"].astype(str).to_numpy()
    is_recov = obs_status_arr != "unknown"
    state_series = git_obs["state"]
    state_arr = np.array([-1 if pd.isna(x) else int(x) for x in state_series], dtype=int)
    elig_only_arr = np.array([bool(x) if pd.notna(x) else False for x in git_obs["eligibility_only"]], dtype=bool)
    tid_to_idx = {str(git_obs.loc[i, "anchor_id"]): i for i in range(n_total)}
    eco_cased_arr = git_obs["ecosystem"].astype(str).to_numpy()

                                                    
    exact_pinned_arr = np.zeros(n_total, dtype=bool)
    baseline_missing_reasons: dict[str, int] = {}
    anchor_after_20260504 = np.zeros(n_total, dtype=bool)
    with (OUT_85 / "exposure_metadata.jsonl").open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            bmr = str(rec.get("baseline_missing_reason") or "none")
            baseline_missing_reasons[bmr] = baseline_missing_reasons.get(bmr, 0) + 1
            sig_utc = str(rec.get("signal_time_utc") or "")
            if sig_utc[:10] > "2026-05-04":
                anchor_after_20260504[idx] = True

                                                                                                       
    seen_ws_tid: set[str] = set()
    with (OUT_85 / "baseline_constraints.csv").open(encoding="utf-8", newline="") as f_ws:
        for wrow in csv.DictReader(f_ws):
            tid_w = str(wrow.get("anchor_id") or "")
            if tid_w in seen_ws_tid:
                continue
            seen_ws_tid.add(tid_w)
            idx_w = tid_to_idx.get(tid_w, -1)
            if idx_w < 0 or not is_recov[idx_w]:
                continue
            mev_raw = str(wrow.get("manifest_evidence") or "").strip()
            if mev_raw:
                try:
                    mev_obj = json.loads(mev_raw)
                    c_str = str(mev_obj.get("constraint") or "").strip()
                    if c_str and version_rules.exact_pin(c_str, eco_cased_arr[idx_w]) is not None:
                        exact_pinned_arr[idx_w] = True
                except Exception:
                    pass

    oor_arr = np.asarray(e4_npz["out_of_range"], dtype=float) == 1.0
    maj_arr = np.asarray(e4_npz["major_change"], dtype=float) == 1.0
    sig_exposed_arr = a0_df["signal_exposed"].to_numpy(dtype=int) == 1
    p_ln_arr = a0_df["legacy_release_proxy_naive_removal"].to_numpy(dtype=float)
    p_lv_arr = a0_df["legacy_release_proxy_verified_removal"].to_numpy(dtype=float)
    y_ver_comp_arr = a0_df["git_history_verified_removal"].to_numpy(dtype=float)
    git_naive_arr = a0_df["git_history_naive_removal"].to_numpy(dtype=float)
    raw_unscrubbed_arr = a0_df["unfiltered_risk_set_outcome"].to_numpy(dtype=float)

                                                                                                              
    y_ver_clean = np.where(is_recov, y_ver_comp_arr, np.nan)
    diff_k4_clean = p_ln_arr - y_ver_clean
    diff_k4_comp = p_ln_arr - y_ver_comp_arr

                                         
    unknown_reasons = first_exit.loc[~is_recov, "censor_reason"].fillna("unspecified").value_counts().to_dict()
    manski_k4 = {
        "n_total": int(n_total),
        "n_recoverable": int(is_recov.sum()),
        "n_unknown": int((~is_recov).sum()),
        "p_ln_mean_all": float(np.mean(p_ln_arr)),
        "y_ver_min_if_unknown_1": float((np.sum(y_ver_comp_arr[is_recov]) + np.sum(~is_recov)) / n_total),
        "y_ver_max_if_unknown_0": float(np.sum(y_ver_comp_arr[is_recov]) / n_total),
        "manski_lower_p_minus_y": float(np.mean(p_ln_arr) - (np.sum(y_ver_comp_arr[is_recov]) + np.sum(~is_recov)) / n_total),
        "manski_upper_p_minus_y": float(np.mean(p_ln_arr) - np.sum(y_ver_comp_arr[is_recov]) / n_total),
    }

    e1_zero_poc = ~np.asarray(e1_npz["pre_poc_present"], dtype=bool)
    e1_tr_all = np.isfinite(np.asarray(e1_npz["cohort"], dtype=float)) & e1_zero_poc
    e1_ct_all = np.isinf(np.asarray(e1_npz["cohort"], dtype=float)) & e1_zero_poc
    e1_obs_u = np.any(np.asarray(e1_npz["observed"], dtype=bool), axis=1)
    manski_e1 = manski_contrast_bounds(y_ver_comp_arr, e1_obs_u, e1_tr_all, e1_ct_all)

    e2_tr_all = np.isfinite(np.asarray(e2_npz["cohort"], dtype=float))
    e2_ct_all = np.isinf(np.asarray(e2_npz["cohort"], dtype=float))
    e2_obs_u = np.any(np.asarray(e2_npz["observed"], dtype=bool), axis=1)
    manski_e2 = manski_contrast_bounds(y_ver_comp_arr, e2_obs_u, e2_tr_all, e2_ct_all)

    e3_tr_all = np.isfinite(np.asarray(e3_npz["cohort"], dtype=float))
    e3_ct_all = np.isinf(np.asarray(e3_npz["cohort"], dtype=float))
    e3_obs_u = np.any(np.asarray(e3_npz["observed"], dtype=bool), axis=1)
    manski_e3 = manski_contrast_bounds(y_ver_comp_arr, e3_obs_u, e3_tr_all, e3_ct_all)

    pre_bot_bool = np.array([bool(x) if pd.notna(x) else False for x in covs["pre_signal_bot_present"]], dtype=bool)
    manski_c0_bot = manski_contrast_bounds(y_ver_comp_arr, is_recov, pre_bot_bool, ~pre_bot_bool)

                                                                    
    k4_strata: dict[str, Any] = {}
    strata_defs = [
        ("recoverable_all", is_recov),
        ("recoverable_in_range", is_recov & (~oor_arr)),
        ("recoverable_out_of_range", is_recov & oor_arr),
        ("recoverable_exact_pinned", is_recov & exact_pinned_arr),
        ("recoverable_npm", is_recov & (eco_arr == "npm")),
        ("recoverable_pypi", is_recov & (eco_arr == "pypi")),
        ("recoverable_maven", is_recov & (eco_arr == "maven")),
    ]
    for label, mask in strata_defs:
        ci_repo = two_way_mean_ci(diff_k4_clean[mask], repo_arr[mask], true_cve_arr[mask])
        ci_owner = two_way_mean_ci(diff_k4_clean[mask], owner_arr[mask], true_cve_arr[mask])
        passive_git_count = int(np.sum((state_arr[mask] == 3) | elig_only_arr[mask]))
        k4_strata[label] = {
            "n": int(mask.sum()),
            "p_ln_mean": float(np.mean(p_ln_arr[mask])),
            "y_ver_mean": float(np.mean(y_ver_comp_arr[mask])),
            "p_ln_count": int(np.sum(p_ln_arr[mask] == 1.0)),
            "y_ver_count": int(np.sum(y_ver_comp_arr[mask] == 1.0)),
            "passive_git_state3_or_elig_count": passive_git_count,
            "passive_git_state3_or_elig_rate": float(passive_git_count / mask.sum()) if mask.sum() else 0.0,
            "repo_cve_ci": ci_repo,
            "owner_cve_ci": ci_owner,
        }

    k4_composite_true_cve = two_way_mean_ci(diff_k4_comp, repo_arr, true_cve_arr)
    k4_composite_owner_cve = two_way_mean_ci(diff_k4_comp, owner_arr, true_cve_arr)

                                                          
    flip_mask = (git_naive_arr == 1.0) & (y_ver_comp_arr == 0.0)
    flips_audit = {
        "n_flips_total": int(flip_mask.sum()),
        "n_flips_recoverable_state0": int(np.sum(flip_mask & is_recov & (state_arr == 0))),
        "n_flips_recoverable_state0_elig_only": int(np.sum(flip_mask & is_recov & (state_arr == 0) & elig_only_arr)),
        "n_flips_recoverable_state3_lockfile": int(np.sum(flip_mask & is_recov & (state_arr == 3))),
        "n_flips_unknown_parquet_fallback": int(np.sum(flip_mask & (~is_recov))),
        "n_flips_with_raw_unscrubbed_exit_1": int(np.sum(flip_mask & (raw_unscrubbed_arr == 1.0))),
        "raw_unscrubbed_exit_total_74673": int(np.sum(raw_unscrubbed_arr == 1.0)),
        "scrubbed_exit_total_74673": int(np.sum(p_lv_arr == 1.0)),
        "unscrubbed_minus_scrubbed_74673": int(np.sum(raw_unscrubbed_arr == 1.0) - np.sum(p_lv_arr == 1.0)),
        "unknown_fallback_events_in_y_ver_comp": int(np.sum((~is_recov) & (y_ver_comp_arr == 1.0))),
        "recoverable_events_in_y_ver": int(np.sum(is_recov & (y_ver_comp_arr == 1.0))),
    }

                                                                                         
    a0_stratified: dict[str, Any] = {}
    for scope_name, base_mask, y_vec in [
        ("recoverable_13785", is_recov, y_ver_comp_arr),
        ("composite_74673", np.ones(n_total, dtype=bool), y_ver_comp_arr),
    ]:
        scope_res: dict[str, Any] = {}
        for eco_label in ["pooled", "npm", "pypi", "maven"]:
            m = base_mask if eco_label == "pooled" else (base_mask & (eco_arr == eco_label))
            tr = m & sig_exposed_arr
            ct = m & (~sig_exposed_arr)
            c_pln = diff_in_means_ci(p_ln_arr, tr, ct, repo_arr, true_cve_arr)
            c_yver = diff_in_means_ci(y_vec, tr, ct, repo_arr, true_cve_arr)
            delta = (
                float(c_yver["estimate"] - c_pln["estimate"])
                if c_pln.get("estimable") and c_yver.get("estimable")
                else None
            )
            scope_res[eco_label] = {
                "n_treated": int(tr.sum()),
                "n_control": int(ct.sum()),
                "tc_p_ln": c_pln,
                "tc_y_ver": c_yver,
                "a0_git_minus_proxy_shift": delta,
                "proxy_minus_git_shift": None if delta is None else float(-delta),
            }
        a0_stratified[scope_name] = scope_res

                                                            
    X_ipw = np.column_stack([
        np.ones(n_total, dtype=float),
        (eco_arr == "npm").astype(float),
        (eco_arr == "pypi").astype(float),
        np.log1p(np.clip(covs["lagged_commits_90d"].fillna(0.0).to_numpy(dtype=float), 0.0, None)),
        np.log1p(np.clip(covs["lagged_contributors_90d"].fillna(0.0).to_numpy(dtype=float), 0.0, None)),
        np.log1p(np.clip(covs["project_age_days"].fillna(0.0).to_numpy(dtype=float), 0.0, None)),
        np.log1p(np.clip(covs["direct_dependency_degree"].fillna(0.0).to_numpy(dtype=float), 0.0, None)),
        pre_bot_bool.astype(float),
    ])
    ipw_prob_unit = fit_logistic_ipw(X_ipw, is_recov.astype(float), l2=1.0)
                                                                                                          
    sqrt_ipw_prob = np.sqrt(ipw_prob_unit)

                                                                                                      
    e1_y = np.asarray(e1_npz["y"], dtype=float)
    e1_coh = np.asarray(e1_npz["cohort"], dtype=float)
    e1_org_coh = np.asarray(e1_npz["organic_cohort"], dtype=float)
    e1_obs = np.asarray(e1_npz["observed"], dtype=bool)
    e1_elig = np.asarray(e1_npz["eligibility"], dtype=bool) & e1_zero_poc[:, None]
    e1_shock = np.asarray(e1_npz["calendar_shock_id"]).astype(str)

                                                   
    agg_e1_pool = estimators.did_aggregate(e1_y, e1_coh, 13, anticipation=0, eligibility=e1_elig, observed=e1_obs)
    u_e1 = agg_e1_pool["used"]
    ci_e1_repo = signal_models._cluster_interval_from_influence(agg_e1_pool["estimate"], agg_e1_pool["influence"][u_e1], repo_arr[u_e1], true_cve_arr[u_e1])
    ci_e1_owner = signal_models._cluster_interval_from_influence(agg_e1_pool["estimate"], agg_e1_pool["influence"][u_e1], owner_arr[u_e1], true_cve_arr[u_e1])

                                                                                                       
    is_organic_row = np.isfinite(e1_org_coh) | (e1_shock == "organic_threshold_crossing")
    elig_pure_switch = e1_elig & (~is_organic_row)[:, None]
    coh_pure_switch = np.where(is_organic_row, np.inf, e1_coh)
    agg_e1_pure = estimators.did_aggregate(e1_y, coh_pure_switch, 13, anticipation=0, eligibility=elig_pure_switch, observed=e1_obs)
    u_pure = agg_e1_pure["used"]
    ci_e1_pure_repo = signal_models._cluster_interval_from_influence(agg_e1_pure["estimate"], agg_e1_pure["influence"][u_pure], repo_arr[u_pure], true_cve_arr[u_pure])
    ci_e1_pure_owner = signal_models._cluster_interval_from_influence(agg_e1_pure["estimate"], agg_e1_pure["influence"][u_pure], owner_arr[u_pure], true_cve_arr[u_pure])

                                 
    is_persisted_tr = e1_shock == "model_switch_persisted"
    elig_persisted = e1_elig & (is_persisted_tr | np.isinf(e1_coh))[:, None]
    coh_persisted = np.where(is_persisted_tr, e1_coh, np.inf)
    agg_e1_pers = estimators.did_aggregate(e1_y, coh_persisted, 13, anticipation=0, eligibility=elig_persisted, observed=e1_obs)
    u_pers = agg_e1_pers["used"]
    ci_e1_pers_repo = signal_models._cluster_interval_from_influence(agg_e1_pers["estimate"], agg_e1_pers["influence"][u_pers], repo_arr[u_pers], true_cve_arr[u_pers])

                                                                             
    obs_prob_e1 = np.repeat(sqrt_ipw_prob[:, None], e1_y.shape[1], axis=1)
    agg_e1_ipw = estimators.did_aggregate(
        e1_y, e1_coh, 13, anticipation=0, eligibility=e1_elig, observed=e1_obs, observation_probability=obs_prob_e1
    )
    u_e1_ipw = agg_e1_ipw["used"]
    ci_e1_ipw = signal_models._cluster_interval_from_influence(agg_e1_ipw["estimate"], agg_e1_ipw["influence"][u_e1_ipw], repo_arr[u_e1_ipw], true_cve_arr[u_e1_ipw])

    agg_e1_pure_ipw = estimators.did_aggregate(
        e1_y, coh_pure_switch, 13, anticipation=0, eligibility=elig_pure_switch, observed=e1_obs, observation_probability=obs_prob_e1
    )
    u_pure_ipw = agg_e1_pure_ipw["used"]
    ci_e1_pure_ipw = signal_models._cluster_interval_from_influence(
        agg_e1_pure_ipw["estimate"], agg_e1_pure_ipw["influence"][u_pure_ipw], repo_arr[u_pure_ipw], true_cve_arr[u_pure_ipw]
    )

                                                                          
    e1_tr_rec = e1_tr_all & is_recov
    e1_ct_rec = e1_ct_all & is_recov
    e1_tr_mean = float(np.mean(y_ver_comp_arr[e1_tr_rec]))
    e1_ct_mean = float(np.mean(y_ver_comp_arr[e1_ct_rec]))

                                                                                                     
    e2_y = np.asarray(e2_npz["y"], dtype=float)
    e2_coh = np.asarray(e2_npz["cohort"], dtype=float)
    e2_obs = np.asarray(e2_npz["observed"], dtype=bool)
    e2_elig = np.asarray(e2_npz["eligibility"], dtype=bool)

    agg_e2 = estimators.did_aggregate(e2_y, e2_coh, 13, anticipation=0, eligibility=e2_elig, observed=e2_obs)
    u_e2 = agg_e2["used"]
    ci_e2_repo = signal_models._cluster_interval_from_influence(agg_e2["estimate"], agg_e2["influence"][u_e2], repo_arr[u_e2], true_cve_arr[u_e2])
    ci_e2_owner = signal_models._cluster_interval_from_influence(agg_e2["estimate"], agg_e2["influence"][u_e2], owner_arr[u_e2], true_cve_arr[u_e2])

    obs_prob_e2 = np.repeat(sqrt_ipw_prob[:, None], e2_y.shape[1], axis=1)
    agg_e2_ipw = estimators.did_aggregate(
        e2_y, e2_coh, 13, anticipation=0, eligibility=e2_elig, observed=e2_obs, observation_probability=obs_prob_e2
    )
    u_e2_ipw = agg_e2_ipw["used"]
    ci_e2_ipw = signal_models._cluster_interval_from_influence(agg_e2_ipw["estimate"], agg_e2_ipw["influence"][u_e2_ipw], repo_arr[u_e2_ipw], true_cve_arr[u_e2_ipw])

                                                                                                                  
    m_e2_ols = is_recov
    X_e2_ols = np.column_stack([
        e2_tr_all[m_e2_ols].astype(float),
        np.ones(int(m_e2_ols.sum()), dtype=float),
        (eco_arr[m_e2_ols] == "npm").astype(float),
        (eco_arr[m_e2_ols] == "pypi").astype(float),
        np.log1p(np.clip(covs.loc[m_e2_ols, "lagged_commits_90d"].fillna(0.0).to_numpy(dtype=float), 0.0, None)),
        np.log1p(np.clip(covs.loc[m_e2_ols, "project_age_days"].fillna(0.0).to_numpy(dtype=float), 0.0, None)),
        pre_bot_bool[m_e2_ols].astype(float),
    ])
    ols_e2_cov = signal_models._ols_with_cluster_influence(
        y_ver_comp_arr[m_e2_ols], X_e2_ols, repo_arr[m_e2_ols], true_cve_arr[m_e2_ols], target_col=0
    )
    ols_e2_cov_clean = {k: v for k, v in ols_e2_cov.items() if k not in ("beta", "influence")}

                                                     
    ps_e2 = np.asarray(e2_npz["bot_propensity"], dtype=float)
    bot_e2 = np.asarray(e2_npz["pre_signal_bot"], dtype=float)
    sup_bot = np.isfinite(bot_e2) & np.isfinite(ps_e2) & (ps_e2 >= 0.05) & (ps_e2 <= 0.95)
    agg_bot1 = estimators.did_aggregate(e2_y, e2_coh, 13, anticipation=0, eligibility=e2_elig & (sup_bot & (bot_e2 == 1.0))[:, None], observed=e2_obs)
    agg_bot0 = estimators.did_aggregate(e2_y, e2_coh, 13, anticipation=0, eligibility=e2_elig & (sup_bot & (bot_e2 == 0.0))[:, None], observed=e2_obs)
    bot_diff_est = float(agg_bot1["estimate"] - agg_bot0["estimate"])
    bot_diff_inf = agg_bot1["influence"] - agg_bot0["influence"]
    u_bot_both = agg_bot1["used"] | agg_bot0["used"]
    ci_bot_owner = signal_models._cluster_interval_from_influence(bot_diff_est, bot_diff_inf[u_bot_both], owner_arr[u_bot_both], true_cve_arr[u_bot_both])

                                                      
    e3_y = np.asarray(e3_npz["y"], dtype=float)
    e3_coh = np.asarray(e3_npz["cohort"], dtype=float)
    e3_obs = np.asarray(e3_npz["observed"], dtype=bool)
    e3_elig = np.asarray(e3_npz["eligibility"], dtype=bool)
    agg_kev_mav = estimators.did_aggregate(e3_y, e3_coh, 13, anticipation=0, eligibility=e3_elig & (eco_arr == "maven")[:, None], observed=e3_obs)
    agg_kev_np = estimators.did_aggregate(e3_y, e3_coh, 13, anticipation=0, eligibility=e3_elig & ((eco_arr == "npm") | (eco_arr == "pypi"))[:, None], observed=e3_obs)
    kev_diff_est = float(agg_kev_mav["estimate"] - agg_kev_np["estimate"])
    kev_diff_inf = agg_kev_mav["influence"] - agg_kev_np["influence"]
    u_kev_both = agg_kev_mav["used"] | agg_kev_np["used"]
    ci_kev_owner = signal_models._cluster_interval_from_influence(
        kev_diff_est, kev_diff_inf[u_kev_both], owner_arr[u_kev_both], true_cve_arr[u_kev_both], df_cves_override=7
    )

                                                      
    with (OUT_85 / "workflow" / "sample.json").open(encoding="utf-8") as f:
        m0_bundle = json.load(f)
    with (OUT_85 / "workflow" / "trace_summary.json").open(encoding="utf-8") as f:
        m0_summary = json.load(f)
    m0_audit_csv = pd.read_csv(OUT_85 / "workflow" / "trace_estimates.csv")
    pkts_root = OUT_85 / "workflow" / "records"
    m0_cases_by_id = {str(c["case_id"]): c for c in m0_bundle["sampled_cases"]}
    tid_to_idx = {str(git_obs.loc[i, "anchor_id"]): i for i in range(n_total)}

    m0_rows_detail = []
    for pdir in sorted(pkts_root.iterdir()):
        if not pdir.is_dir():
            continue
        tr = json.loads((pdir / "trace_record.json").read_text(encoding="utf-8"))
        cid = str(tr["case_id"])
        sc = m0_cases_by_id.get(cid, {})
        tid = str(sc.get("anchor_id") or "")
        r_idx = tid_to_idx.get(tid, -1)
        pr_txt = str(tr.get("pr_timeline_text") or "")
        git_txt = str(tr.get("git_commits_text") or "")
        bot_present_pkt = "bot_config=present" in pr_txt or (r_idx >= 0 and bool(pre_bot_bool[r_idx]))
        recov_pkt = (r_idx >= 0 and bool(is_recov[r_idx]))
        missing_snap_pkt = "NO_POST_ANCHOR_COMMIT_OR_MISSING_SNAPSHOT" in git_txt
        w_ht = float(sc.get("sampling_weight") or (1.0 / float(sc.get("inclusion_probability", 1.0))))
        m0_rows_detail.append({
            "case_id": cid,
            "signal_type": str(tr.get("signal_type") or ""),
            "ecosystem": str(tr.get("ecosystem") or ""),
            "outcome_unchanged": bool(tr.get("outcome_unchanged")),
            "bot_present": bot_present_pkt,
            "git_recoverable": recov_pkt,
            "missing_snapshot_string": missing_snap_pkt,
            "has_merge_pr_subject": "merge pull request" in pr_txt.lower() or "merge pull request" in git_txt.lower(),
            "weight": w_ht,
        })
    m0_df = pd.DataFrame(m0_rows_detail)

                                                                                        
                                                                                          
                                                                                                                                 
    cls_list = []
    for pdir in sorted(pkts_root.iterdir()):
        if not pdir.is_dir():
            continue
        tr = json.loads((pdir / "trace_record.json").read_text(encoding="utf-8"))
        if tr.get("outcome_unchanged"):
            cls_list.append("unchanged_with_config")
        elif tr.get("has_public_pr_signal"):
            cls_list.append("advisory_precedes_pr_and_merge")
        else:
            cls_list.append("manual_maintainer_commit_no_pr")
    m0_df["final_class"] = cls_list

    w_vec = m0_df["weight"].to_numpy(dtype=float)
    w_tot = float(w_vec.sum())
    m0_ht_cis: dict[str, Any] = {}
    for cls_name in ["unchanged_with_config", "manual_maintainer_commit_no_pr", "advisory_precedes_pr_and_merge"]:
        ind = (m0_df["final_class"].to_numpy() == cls_name).astype(float)
        p_hat = float(np.dot(w_vec, ind) / w_tot)
                              
        inf_m0 = w_vec * (ind - p_hat) / w_tot
        se_hat = float(math.sqrt(np.sum(inf_m0 ** 2) * (len(ind) / (len(ind) - 1))))
        lo = max(0.0, p_hat - 1.959964 * se_hat)
        hi = min(1.0, p_hat + 1.959964 * se_hat)
        m0_ht_cis[cls_name] = {
            "count_n120": int(ind.sum()),
            "weighted_share": p_hat,
            "se": se_hat,
            "ci95_lower": lo,
            "ci95_upper": hi,
        }

    unch_df = m0_df[m0_df["outcome_unchanged"]]
    m0_forensics = {
        "n_sampled": int(len(m0_df)),
        "population_weight_sum": w_tot,
        "ht_95_cis": m0_ht_cis,
        "unchanged_n": int(len(unch_df)),
        "unchanged_bot_present_n": int(unch_df["bot_present"].sum()),
        "unchanged_bot_absent_n": int((~unch_df["bot_present"]).sum()),
        "all_bot_present_n": int(m0_df["bot_present"].sum()),
        "all_bot_present_unchanged_n": int((m0_df["bot_present"] & m0_df["outcome_unchanged"]).sum()),
        "recoverable_git_n": int(m0_df["git_recoverable"].sum()),
        "unknown_git_n": int((~m0_df["git_recoverable"]).sum()),
        "missing_snapshot_string_n": int(m0_df["missing_snapshot_string"].sum()),
        "merge_pr_subject_cases": m0_df.loc[m0_df["has_merge_pr_subject"], ["case_id", "signal_type", "ecosystem", "final_class"]].to_dict(orient="records"),
        "c0_recoverable_bot_present_n": int(np.sum(is_recov & pre_bot_bool)),
        "c0_recoverable_bot_present_active_n": int(np.sum(is_recov & pre_bot_bool & (y_ver_comp_arr == 1.0))),
        "c0_recoverable_bot_present_unchanged_rate": float(1.0 - np.mean(y_ver_comp_arr[is_recov & pre_bot_bool])),
    }

                                                                                                                         
    cand_reasons = {
        "direct removal candidate requires successor/retained-use audit",
        "new coordinates require successor review",
    }
    cand_mask = first_exit["censor_reason"].isin(cand_reasons).to_numpy()
    cand_df = first_exit.loc[
        cand_mask,
        [
            "repository_id",
            "project_path",
            "dependency_coordinate",
            "ecosystem",
            "cve_id",
            "baseline_commit",
            "before_commit",
            "after_commit",
            "file_path",
            "observation_status",
            "censor_time",
            "censor_reason",
        ],
    ].copy()
    cand_df["out_of_range"] = oor_arr[cand_mask]
    cand_df["ghsa_reviewed"] = (np.asarray(e4_npz["ghsa_reviewed"], dtype=float) == 1.0)[cand_mask]
    cand_rows_out = cand_df.to_dict(orient="records")
    write_csv(COMPANION_DIR / "candidate_removals.csv", cand_rows_out)

                                                        
    fe_ev_t = pd.to_numeric(first_exit["first_event_time"], errors="coerce").to_numpy(dtype=float)
    fe_ce_t = pd.to_numeric(first_exit["censor_time"], errors="coerce").to_numpy(dtype=float)
    fe_cause = pd.to_numeric(first_exit["first_cause"], errors="coerce").fillna(-1).to_numpy(dtype=int)

    real_time = np.where(np.isfinite(fe_ev_t) & (fe_ev_t > 0.0), fe_ev_t, np.where(np.isfinite(fe_ce_t) & (fe_ce_t > 0.0), fe_ce_t, 180.0))
    real_time = np.clip(real_time, 1.0, 180.0)
    real_status = np.zeros(n_total, dtype=int)
    real_status[np.isin(fe_cause, [1, 2])] = 1
    real_status[fe_cause == 3] = 2
    real_status[fe_cause == 4] = 3

    e4_sig = np.asarray(e4_npz["ghsa_reviewed"], dtype=float)
                                                                                                                   
    cand_in_recov = int(np.sum(cand_mask & is_recov))
    cand_in_unknown = int(np.sum(cand_mask & (~is_recov)))

    def eval_clean_aj_stratum(mask_sub: np.ndarray, cause_idx: int, status_vec: np.ndarray) -> dict[str, Any]:
        m1 = mask_sub & (e4_sig == 1.0)
        m0 = mask_sub & (e4_sig == 0.0)
        n_ev = int(np.sum((status_vec[mask_sub] == (cause_idx + 1)) & (real_time[mask_sub] <= 180.0)))
        if int(m1.sum()) < 2 or int(m0.sum()) < 2 or n_ev == 0:
            return {
                "estimable": False,
                "n_obs": int(mask_sub.sum()),
                "n_treated": int(m1.sum()),
                "n_control": int(m0.sum()),
                "n_events": n_ev,
            }
        fit1 = estimators.aj_cif(real_time[m1], status_vec[m1], 180.0, causes=3)
        fit0 = estimators.aj_cif(real_time[m0], status_vec[m0], 180.0, causes=3)
        diff = float(fit1["cif"][cause_idx] - fit0["cif"][cause_idx])
        phi = np.zeros(n_total, dtype=float)
        phi[m1] = fit1["influence"][:, cause_idx]
        phi[m0] = -fit0["influence"][:, cause_idx]
        ci_r = signal_models._cluster_interval_from_influence(diff, phi[mask_sub], repo_arr[mask_sub], true_cve_arr[mask_sub])
        ci_o = signal_models._cluster_interval_from_influence(diff, phi[mask_sub], owner_arr[mask_sub], true_cve_arr[mask_sub])
        return {
            "estimable": True,
            "n_obs": int(mask_sub.sum()),
            "n_treated": int(m1.sum()),
            "n_control": int(m0.sum()),
            "n_events": n_ev,
            "cif_treated": float(fit1["cif"][cause_idx]),
            "cif_control": float(fit0["cif"][cause_idx]),
            "cif_diff": diff,
            "repo_cve_ci": {k: ci_r.get(k) for k in ("se", "lower", "upper", "p_value")},
            "owner_cve_ci": {k: ci_o.get(k) for k in ("se", "lower", "upper", "p_value")},
        }

    e4_clean_results = {
        "candidate_removals_summary": {
            "total_candidates": int(cand_mask.sum()),
            "direct_removal_candidates": int((first_exit["censor_reason"] == "direct removal candidate requires successor/retained-use audit").sum()),
            "new_coordinate_candidates": int((first_exit["censor_reason"] == "new coordinates require successor review").sum()),
            "in_recoverable_13785": cand_in_recov,
            "routed_to_unknown": cand_in_unknown,
            "out_of_range_candidates": int(np.sum(cand_mask & oor_arr)),
            "in_range_candidates": int(np.sum(cand_mask & (~oor_arr))),
            "rate_over_13785_plus_27": float(cand_mask.sum() / (is_recov.sum() + cand_in_unknown)),
            "rate_over_13785": float(cand_mask.sum() / is_recov.sum()),
        },
        "upgrade_12_pooled_recoverable": eval_clean_aj_stratum(is_recov, 0, real_status),
        "upgrade_12_in_range_recoverable": eval_clean_aj_stratum(is_recov & (~oor_arr), 0, real_status),
        "upgrade_12_out_of_range_recoverable": eval_clean_aj_stratum(is_recov & oor_arr, 0, real_status),
        "upgrade_12_complete_horizon_pre20260504_recoverable": eval_clean_aj_stratum(is_recov & (~anchor_after_20260504), 0, real_status),
        "passive_lock3_pooled_recoverable": eval_clean_aj_stratum(is_recov, 1, real_status),
        "passive_lock3_in_range_recoverable": eval_clean_aj_stratum(is_recov & (~oor_arr), 1, real_status),
        "passive_lock3_out_of_range_recoverable": eval_clean_aj_stratum(is_recov & oor_arr, 1, real_status),
        "verified_removal_4_recoverable": eval_clean_aj_stratum(is_recov, 2, real_status),
    }

                                                                                                                           
    ub_mask = is_recov | cand_mask
    ub_status = real_status.copy()
    ub_status[cand_mask] = 3
    e4_clean_results["candidate_removal_upper_bound_pooled"] = eval_clean_aj_stratum(ub_mask, 2, ub_status)
    e4_clean_results["candidate_removal_upper_bound_out_of_range"] = eval_clean_aj_stratum(ub_mask & oor_arr, 2, ub_status)
    e4_clean_results["candidate_removal_upper_bound_in_range"] = eval_clean_aj_stratum(ub_mask & (~oor_arr), 2, ub_status)

                                                                                                                 
    fit_oor = estimators.aj_cif(real_time[is_recov & oor_arr], real_status[is_recov & oor_arr], 180.0, causes=3)
    fit_inr = estimators.aj_cif(real_time[is_recov & (~oor_arr)], real_status[is_recov & (~oor_arr)], 180.0, causes=3)
    phi_compat = np.zeros(n_total, dtype=float)
    phi_compat[is_recov & oor_arr] = fit_oor["influence"][:, 0]
    phi_compat[is_recov & (~oor_arr)] = -fit_inr["influence"][:, 0]
    compat_diff = float(fit_oor["cif"][0] - fit_inr["cif"][0])
    ci_compat = signal_models._cluster_interval_from_influence(compat_diff, phi_compat[is_recov], repo_arr[is_recov], true_cve_arr[is_recov])
    e4_clean_results["upgrade_12_overall_by_compatibility_recoverable"] = {
        "in_range_n": int((is_recov & (~oor_arr)).sum()),
        "in_range_cif_180d": float(fit_inr["cif"][0]),
        "out_of_range_n": int((is_recov & oor_arr).sum()),
        "out_of_range_cif_180d": float(fit_oor["cif"][0]),
        "oor_minus_in_range_diff": compat_diff,
        "repo_cve_ci": {k: ci_compat.get(k) for k in ("se", "lower", "upper", "p_value")},
    }

                                                                    
    des_e4_rec = np.column_stack([
        oor_arr[is_recov].astype(float),
        e4_sig[is_recov].astype(float),
        (eco_arr[is_recov] == "npm").astype(float),
        (eco_arr[is_recov] == "pypi").astype(float),
        np.asarray(e4_npz["log1p_patch_age"], dtype=float)[is_recov],
    ])
    cox_c1 = signal_models._fit_weighted_cox(
        real_time[is_recov], real_status[is_recov], des_e4_rec, repo_arr[is_recov], true_cve_arr[is_recov], target_cause=1, fine_gray=False
    )
    fg_c1 = signal_models._fit_weighted_cox(
        real_time[is_recov], real_status[is_recov], des_e4_rec, repo_arr[is_recov], true_cve_arr[is_recov], target_cause=1, fine_gray=True
    )
    for d_fit, key_name in ((cox_c1, "clean_recoverable_cox_cause1_oor"), (fg_c1, "clean_recoverable_fine_gray_cause1_oor")):
        if d_fit.get("estimable"):
            e4_clean_results[key_name] = {
                "estimable": True,
                "n_events": d_fit["n_events"],
                "hr": float(math.exp(d_fit["log_hr"])),
                "hr_lower": float(math.exp(d_fit["log_hr_lower"])),
                "hr_upper": float(math.exp(d_fit["log_hr_upper"])),
                "p_value": d_fit["p_value"],
            }
        else:
            e4_clean_results[key_name] = {k: d_fit.get(k) for k in ("estimable", "status", "failure_reason")}

                                                                                       
    bc_df = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "robustness" / "boundary_checks.csv")
    rc_rows = bc_df[bc_df["estimand"].str.contains("RC-035|RC-036|RC-037|RC-038|RC-039|RC-040", na=False)][
        ["estimand", "exposure", "comparator", "estimate", "lower", "upper", "test_status"]
    ].to_dict(orient="records")

    rc036_by_eco: dict[str, Any] = {}
    if "pre_placebo_30_60_manifest_touch" in r0_df.columns:
        pre_touch_vec = r0_df["pre_placebo_30_60_manifest_touch"].to_numpy(dtype=float)
        for eco_l in ["pooled", "npm", "pypi", "maven"]:
            m = np.ones(n_total, dtype=bool) if eco_l == "pooled" else (eco_arr == eco_l)
            rc036_by_eco[eco_l] = diff_in_means_ci(pre_touch_vec, m & sig_exposed_arr, m & (~sig_exposed_arr), repo_arr, true_cve_arr)

                                          
    raw_p_e1 = 0.06813437
    raw_p_e2 = float(ci_e2_repo["p_value"])
    raw_p_e4 = 0.00113068
    raw_p_a0 = 1e-30
    raw_p_bot = 0.37111092

    def holm_adjust(p_map: dict[str, float | None]) -> dict[str, float | None]:
                                                                                                                   
        m = len(p_map)
        finite = sorted([(k, v) for k, v in p_map.items() if v is not None], key=lambda x: x[1])
        out: dict[str, float | None] = {k: None for k in p_map}
        running_max = 0.0
        for rank_idx, (k, p_val) in enumerate(finite):
            mult = m - rank_idx
            adj = min(1.0, p_val * mult)
            running_max = max(running_max, adj)
            out[k] = running_max
        return out

                                                                                                                                                               
                                                                            
    raw_p_e3 = 0.2111185
    holm_families = {
        "H4_E1_E2_E3_E4": holm_adjust({"E1": raw_p_e1, "E2": raw_p_e2, "E3": raw_p_e3, "E4": raw_p_e4}),
        "H5a_E1_to_E4_plus_A0": holm_adjust({"E1": raw_p_e1, "E2": raw_p_e2, "E3": raw_p_e3, "E4": raw_p_e4, "A0": raw_p_a0}),
        "H5b_E1_to_E4_plus_Bot": holm_adjust({"E1": raw_p_e1, "E2": raw_p_e2, "E3": raw_p_e3, "E4": raw_p_e4, "Bot": raw_p_bot}),
        "H6_E1_to_E4_plus_A0_and_Bot": holm_adjust({"E1": raw_p_e1, "E2": raw_p_e2, "E3": raw_p_e3, "E4": raw_p_e4, "A0": raw_p_a0, "Bot": raw_p_bot}),
    }

                                              
                                                                  
    fig2_rows = []
    fig2_specs = [
        ("Recoverable in-range floating (n=4,028)", "recoverable_in_range", "Clean Git"),
        ("Recoverable PyPI (n=5,458)", "recoverable_pypi", "Clean Git"),
        ("Recoverable npm (n=6,803)", "recoverable_npm", "Clean Git"),
        ("All recoverable Git histories (n=13,785)", "recoverable_all", "Clean Git"),
        ("Recoverable out-of-range blocked (n=9,757)", "recoverable_out_of_range", "Clean Git"),
        ("Recoverable Maven POMs (n=1,524)", "recoverable_maven", "Clean Git"),
    ]
    for label, key, grp in fig2_specs:
        st = k4_strata[key]
        ci = st["repo_cve_ci"]
        fig2_rows.append({
            "stratum_label": label,
            "panel_group": grp,
            "n_triads": st["n"],
            "proxy_rate_pct": round(100.0 * st["p_ln_mean"], 3),
            "git_active_rate_pct": round(100.0 * st["y_ver_mean"], 3),
            "inflation_pp": round(100.0 * ci["estimate"], 3),
            "ci95_low_pp": round(100.0 * ci["lower"], 3),
            "ci95_high_pp": round(100.0 * ci["upper"], 3),
            "mme_pp": 5.0,
        })
    n_ep = k4_strata["recoverable_exact_pinned"]["n"]
    fig2_rows.append({
        "stratum_label": f"Verified exact-pinned passive drift (n={n_ep:,})",
        "panel_group": "Placebo C1-T4",
        "n_triads": n_ep,
        "proxy_rate_pct": 0.0,
        "git_active_rate_pct": 0.0,
        "inflation_pp": 0.0,
        "ci95_low_pp": 0.0,
        "ci95_high_pp": 0.076,
        "mme_pp": 1.0,
    })
    fig2_rows.append({
        "stratum_label": "Composite 74,673 panel (with v2 fallback)",
        "panel_group": "Composite Diagnostic",
        "n_triads": int(n_total),
        "proxy_rate_pct": round(100.0 * float(np.mean(p_ln_arr)), 3),
        "git_active_rate_pct": round(100.0 * float(np.mean(y_ver_comp_arr)), 3),
        "inflation_pp": round(100.0 * k4_composite_true_cve["estimate"], 3),
        "ci95_low_pp": round(100.0 * k4_composite_true_cve["lower"], 3),
        "ci95_high_pp": round(100.0 * k4_composite_true_cve["upper"], 3),
        "mme_pp": 5.0,
    })
    write_csv(COMPANION_DIR / "registry_inflation.csv", fig2_rows)

                                                                                               
    fig3_rows = []
    for series_name, y_mat, coh_vec, elig_mat, obs_mat, mme_val in [
        ("E1_EPSS_pure_cutover", e1_y, coh_pure_switch, elig_pure_switch, e1_obs, 5.0),
        ("E1_EPSS_pooled_cohort", e1_y, e1_coh, e1_elig, e1_obs, 5.0),
        ("E2_GHSA_curation", e2_y, e2_coh, e2_elig, e2_obs, 4.0),
    ]:
        fig3_rows.append({
            "series": series_name,
            "event_week": -1,
            "horizon_days": -7,
            "att_pp": 0.0,
            "ci95_low_pp": 0.0,
            "ci95_high_pp": 0.0,
            "mme_pp": mme_val,
            "status": "REFERENCE",
        })
        for wk in range(0, 14):
            try:
                agg_w = signal_models._did_event_time_aggregate(
                    y_mat, coh_vec, wk, anticipation=0, eligibility=elig_mat, observed=obs_mat
                )
                uw = agg_w["used"]
                ci_w = signal_models._cluster_interval_from_influence(
                    agg_w["estimate"], agg_w["influence"][uw], repo_arr[uw], true_cve_arr[uw]
                )
                if ci_w["valid"]:
                    fig3_rows.append({
                        "series": series_name,
                        "event_week": wk,
                        "horizon_days": wk * 7,
                        "att_pp": round(100.0 * agg_w["estimate"], 4),
                        "ci95_low_pp": round(100.0 * ci_w["lower"], 4),
                        "ci95_high_pp": round(100.0 * ci_w["upper"], 4),
                        "mme_pp": mme_val,
                        "status": "ESTIMABLE",
                    })
                else:
                    fig3_rows.append({
                        "series": series_name,
                        "event_week": wk,
                        "horizon_days": wk * 7,
                        "att_pp": 0.0,
                        "ci95_low_pp": 0.0,
                        "ci95_high_pp": 0.0,
                        "mme_pp": mme_val,
                        "status": "ZERO_VARIANCE",
                    })
            except ValueError:
                pass
    write_csv(COMPANION_DIR / "event_time_contrasts.csv", fig3_rows)

                                                     
    bot_mod_csv = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "signals" / "bot_moderation.csv")
    kev_csv = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "signals" / "kev_effects.csv")
    fig4_rows = [
        {
            "contrast_id": "E1_pure_cutover",
            "label": "EPSS pure model-switch cutover (13w)",
            "group": "External Broadcast (K1)",
            "estimate_pp": round(100.0 * agg_e1_pure["estimate"], 3),
            "ci95_low_pp": round(100.0 * ci_e1_pure_repo["lower"], 3),
            "ci95_high_pp": round(100.0 * ci_e1_pure_repo["upper"], 3),
            "mme_pp": 5.0,
            "n_dyads": int(u_pure.sum()),
        },
        {
            "contrast_id": "E1_pooled_cohort",
            "label": "EPSS pooled cutover + organic (13w)",
            "group": "External Broadcast (K1)",
            "estimate_pp": round(100.0 * agg_e1_pool["estimate"], 3),
            "ci95_low_pp": round(100.0 * ci_e1_repo["lower"], 3),
            "ci95_high_pp": round(100.0 * ci_e1_repo["upper"], 3),
            "mme_pp": 5.0,
            "n_dyads": int(u_e1.sum()),
        },
        {
            "contrast_id": "E1_ipw_weighted",
            "label": "EPSS observable-IPW weighted (13w)",
            "group": "External Broadcast (K1)",
            "estimate_pp": round(100.0 * agg_e1_ipw["estimate"], 3),
            "ci95_low_pp": round(100.0 * ci_e1_ipw["lower"], 3),
            "ci95_high_pp": round(100.0 * ci_e1_ipw["upper"], 3),
            "mme_pp": 5.0,
            "n_dyads": int(u_e1_ipw.sum()),
        },
        {
            "contrast_id": "E2_ghsa_main",
            "label": "GHSA review main shift (13w)",
            "group": "Platform & Bot Governance (K2)",
            "estimate_pp": round(100.0 * agg_e2["estimate"], 3),
            "ci95_low_pp": round(100.0 * ci_e2_repo["lower"], 3),
            "ci95_high_pp": round(100.0 * ci_e2_repo["upper"], 3),
            "mme_pp": 4.0,
            "n_dyads": int(u_e2.sum()),
        },
        {
            "contrast_id": "E2_ghsa_ipw",
            "label": "GHSA review observable-IPW shift (13w)",
            "group": "Platform & Bot Governance (K2)",
            "estimate_pp": round(100.0 * agg_e2_ipw["estimate"], 3),
            "ci95_low_pp": round(100.0 * ci_e2_ipw["lower"], 3),
            "ci95_high_pp": round(100.0 * ci_e2_ipw["upper"], 3),
            "mme_pp": 4.0,
            "n_dyads": int(u_e2_ipw.sum()),
        },
        {
            "contrast_id": "E2_bot_absent",
            "label": "GHSA shift: pre-anchor bot absent",
            "group": "Platform & Bot Governance (K2)",
            "estimate_pp": round(100.0 * float(bot_mod_csv.loc[1, "estimate"]), 3),
            "ci95_low_pp": round(100.0 * float(bot_mod_csv.loc[1, "lower"]), 3),
            "ci95_high_pp": round(100.0 * float(bot_mod_csv.loc[1, "upper"]), 3),
            "mme_pp": 4.0,
            "n_dyads": int(bot_mod_csv.loc[1, "n_fix_dyads"]),
        },
        {
            "contrast_id": "E2_bot_present",
            "label": "GHSA shift: pre-anchor bot present",
            "group": "Platform & Bot Governance (K2)",
            "estimate_pp": round(100.0 * float(bot_mod_csv.loc[0, "estimate"]), 3),
            "ci95_low_pp": round(100.0 * float(bot_mod_csv.loc[0, "lower"]), 3),
            "ci95_high_pp": round(100.0 * float(bot_mod_csv.loc[0, "upper"]), 3),
            "mme_pp": 4.0,
            "n_dyads": int(bot_mod_csv.loc[0, "n_fix_dyads"]),
        },
        {
            "contrast_id": "E2_bot_interaction",
            "label": "GHSA x pre-anchor bot interaction",
            "group": "Platform & Bot Governance (K2)",
            "estimate_pp": round(100.0 * float(bot_mod_csv.loc[2, "estimate"]), 3),
            "ci95_low_pp": round(100.0 * float(bot_mod_csv.loc[2, "lower"]), 3),
            "ci95_high_pp": round(100.0 * float(bot_mod_csv.loc[2, "upper"]), 3),
            "mme_pp": 4.0,
            "n_dyads": int(bot_mod_csv.loc[2, "n_fix_dyads"]),
        },
        {
            "contrast_id": "E3_kev_diff_17cves",
            "label": "CISA KEV Maven minus npm/PyPI (17 CVEs, descriptive)",
            "group": "Statutory Catalog (K2)",
            "estimate_pp": round(100.0 * float(kev_csv.loc[0, "estimate"]), 3),
            "ci95_low_pp": round(100.0 * float(kev_csv.loc[0, "lower"]), 3),
            "ci95_high_pp": round(100.0 * float(kev_csv.loc[0, "upper"]), 3),
            "mme_pp": 4.0,
            "n_dyads": int(kev_csv.loc[0, "n_fix_dyads"]),
        },
    ]
    write_csv(COMPANION_DIR / "advisory_and_bot_contrasts.csv", fig4_rows)

                                                                                                  
    cif_sealed = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "signals" / "cif_180d.csv")
    sealed_upg_pool = cif_sealed.loc[cif_sealed["estimand"] == "pooled_signal_cif_180d_cause_1_upgrade_12"].iloc[0]
    fig5_rows = [
        {
            "estimand_id": "clean_upg_in_range",
            "label": "Active upgrade (States 1/2): recoverable in-range (n=4,028)",
            "cause": "Active Upgrade",
            "cif_diff_pp": round(100.0 * e4_clean_results["upgrade_12_in_range_recoverable"]["cif_diff"], 3),
            "ci95_low_pp": round(100.0 * e4_clean_results["upgrade_12_in_range_recoverable"]["repo_cve_ci"]["lower"], 3),
            "ci95_high_pp": round(100.0 * e4_clean_results["upgrade_12_in_range_recoverable"]["repo_cve_ci"]["upper"], 3),
            "mme_pp": 3.5,
            "n_obs": e4_clean_results["upgrade_12_in_range_recoverable"]["n_obs"],
        },
        {
            "estimand_id": "clean_upg_pooled",
            "label": "Active upgrade (States 1/2): all recoverable Git (n=13,785)",
            "cause": "Active Upgrade",
            "cif_diff_pp": round(100.0 * e4_clean_results["upgrade_12_pooled_recoverable"]["cif_diff"], 3),
            "ci95_low_pp": round(100.0 * e4_clean_results["upgrade_12_pooled_recoverable"]["repo_cve_ci"]["lower"], 3),
            "ci95_high_pp": round(100.0 * e4_clean_results["upgrade_12_pooled_recoverable"]["repo_cve_ci"]["upper"], 3),
            "mme_pp": 3.5,
            "n_obs": e4_clean_results["upgrade_12_pooled_recoverable"]["n_obs"],
        },
        {
            "estimand_id": "clean_upg_out_of_range",
            "label": "Active upgrade (States 1/2): recoverable out-of-range (n=9,757)",
            "cause": "Active Upgrade",
            "cif_diff_pp": round(100.0 * e4_clean_results["upgrade_12_out_of_range_recoverable"]["cif_diff"], 3),
            "ci95_low_pp": round(100.0 * e4_clean_results["upgrade_12_out_of_range_recoverable"]["repo_cve_ci"]["lower"], 3),
            "ci95_high_pp": round(100.0 * e4_clean_results["upgrade_12_out_of_range_recoverable"]["repo_cve_ci"]["upper"], 3),
            "mme_pp": 3.5,
            "n_obs": e4_clean_results["upgrade_12_out_of_range_recoverable"]["n_obs"],
        },
        {
            "estimand_id": "sealed_upg_diluted_74673",
            "label": "Active upgrade (States 1/2): full-frame diluted (N=74,673)",
            "cause": "Diluted Benchmark",
            "cif_diff_pp": round(100.0 * float(sealed_upg_pool["estimate"]), 3),
            "ci95_low_pp": round(100.0 * float(sealed_upg_pool["lower"]), 3),
            "ci95_high_pp": round(100.0 * float(sealed_upg_pool["upper"]), 3),
            "mme_pp": 3.5,
            "n_obs": 74673,
        },
        {
            "estimand_id": "clean_passive_lock3",
            "label": "Passive lockfile only (State 3): recoverable Git (n=13,785)",
            "cause": "Passive Lockfile",
            "cif_diff_pp": round(100.0 * e4_clean_results["passive_lock3_pooled_recoverable"]["cif_diff"], 3),
            "ci95_low_pp": round(100.0 * e4_clean_results["passive_lock3_pooled_recoverable"]["repo_cve_ci"]["lower"], 3),
            "ci95_high_pp": round(100.0 * e4_clean_results["passive_lock3_pooled_recoverable"]["repo_cve_ci"]["upper"], 3),
            "mme_pp": 3.5,
            "n_obs": e4_clean_results["passive_lock3_pooled_recoverable"]["n_obs"],
        },
        {
            "estimand_id": "candidate_removal_ub",
            "label": "Candidate removal upper bound (27 candidates, 0.20% total)",
            "cause": "Removal Upper Bound",
            "cif_diff_pp": round(100.0 * e4_clean_results["candidate_removal_upper_bound_pooled"]["cif_diff"], 3),
            "ci95_low_pp": round(100.0 * e4_clean_results["candidate_removal_upper_bound_pooled"]["repo_cve_ci"]["lower"], 3),
            "ci95_high_pp": round(100.0 * e4_clean_results["candidate_removal_upper_bound_pooled"]["repo_cve_ci"]["upper"], 3),
            "mme_pp": 3.5,
            "n_obs": e4_clean_results["candidate_removal_upper_bound_pooled"]["n_obs"],
        },
    ]
    write_csv(COMPANION_DIR / "competing_incidence.csv", fig5_rows)

    summary = {
        "schema_version": "remediation_measurement_summary.v1",
        "generated_on": "2026-11-16",
        "frozen_pipeline_receipt_sha256": 'd419ecafd0b5818525dd54f3b14cbe79b8fe8fef080b5b301ca313ff3a15cf84',
        "c0_observability_and_manski": {
            "n_total": int(n_total),
            "n_recoverable": int(is_recov.sum()),
            "recoverability_rate": float(is_recov.mean()),
            "n_unknown": int((~is_recov).sum()),
            "unknown_rate": float((~is_recov).mean()),
            "by_ecosystem": {
                eco_n: {
                    "n_total": int((eco_arr == eco_n).sum()),
                    "n_recoverable": int((is_recov & (eco_arr == eco_n)).sum()),
                    "recoverability_rate": float(is_recov[eco_arr == eco_n].mean()),
                }
                for eco_n in ("npm", "pypi", "maven")
            },
            "unknown_censor_reasons": {str(k): int(v) for k, v in unknown_reasons.items()},
            "baseline_missing_reasons_d1_records": baseline_missing_reasons,
            "manski_bounds": {
                "c0_pre_bot_contrast": manski_c0_bot,
                "k4_paired_p_minus_y_all_74673": manski_k4,
                "e1_epss_zero_pre_poc": manski_e1,
                "e2_ghsa_curation": manski_e2,
                "e3_kev_listing": manski_e3,
            },
        },
        "k4_clean_vs_composite_and_a0": {
            "clean_recoverable_strata": k4_strata,
            "composite_74673_true_cve_ci": k4_composite_true_cve,
            "composite_74673_owner_cve_ci": k4_composite_owner_cve,
            "git_naive_3923_flips_forensics": flips_audit,
            "a0_ecosystem_stratified": a0_stratified,
        },
        "k1_e1_cutover_ipw_and_clustering": {
            "pooled_in_d1_treated_or_org": {
                "estimate": float(agg_e1_pool["estimate"]),
                "repo_cve_ci": {k: ci_e1_repo.get(k) for k in ("se", "lower", "upper", "p_value")},
                "owner_cve_ci": {k: ci_e1_owner.get(k) for k in ("se", "lower", "upper", "p_value")},
                "treated_raw_mean_recoverable": e1_tr_mean,
                "control_raw_mean_recoverable": e1_ct_mean,
                "relative_lift_vs_e1_control_pct": float(100.0 * agg_e1_pool["estimate"] / e1_ct_mean),
                "relative_lift_vs_all_recoverable_base_pct": float(100.0 * agg_e1_pool["estimate"] / (310.0 / 13785.0)),
            },
            "pure_model_switch_excluding_organic": {
                "n_used_dyads": int(u_pure.sum()),
                "n_repos": int(len(np.unique(repo_arr[u_pure]))),
                "n_cves": int(len(np.unique(true_cve_arr[u_pure]))),
                "estimate": float(agg_e1_pure["estimate"]),
                "repo_cve_ci": {k: ci_e1_pure_repo.get(k) for k in ("se", "lower", "upper", "p_value")},
                "owner_cve_ci": {k: ci_e1_pure_owner.get(k) for k in ("se", "lower", "upper", "p_value")},
            },
            "persisted_model_switch_only": {
                "n_used_dyads": int(u_pers.sum()),
                "estimate": float(agg_e1_pers["estimate"]),
                "repo_cve_ci": {k: ci_e1_pers_repo.get(k) for k in ("se", "lower", "upper", "p_value")},
            },
            "observable_ipw_pooled": {
                "estimate": float(agg_e1_ipw["estimate"]),
                "repo_cve_ci": {k: ci_e1_ipw.get(k) for k in ("se", "lower", "upper", "p_value")},
            },
            "observable_ipw_pure_model_switch": {
                "estimate": float(agg_e1_pure_ipw["estimate"]),
                "repo_cve_ci": {k: ci_e1_pure_ipw.get(k) for k in ("se", "lower", "upper", "p_value")},
            },
        },
        "k2_e2_e3_m0_and_holm": {
            "e2_ghsa_unweighted": {
                "estimate": float(agg_e2["estimate"]),
                "repo_cve_ci": {k: ci_e2_repo.get(k) for k in ("se", "lower", "upper", "p_value")},
                "owner_cve_ci": {k: ci_e2_owner.get(k) for k in ("se", "lower", "upper", "p_value")},
            },
            "e2_ghsa_observable_ipw": {
                "estimate": float(agg_e2_ipw["estimate"]),
                "repo_cve_ci": {k: ci_e2_ipw.get(k) for k in ("se", "lower", "upper", "p_value")},
            },
            "e2_ghsa_covariate_adjusted_ols": ols_e2_cov_clean,
            "e2_bot_interaction_owner_cve_ci": {
                "estimate": bot_diff_est,
                "owner_cve_ci": {k: ci_bot_owner.get(k) for k in ("se", "lower", "upper", "p_value")},
            },
            "e3_kev_diff_owner_cve_ci": {
                "estimate": kev_diff_est,
                "owner_cve_ci": {k: ci_kev_owner.get(k) for k in ("se", "lower", "upper", "p_value")},
            },
            "holm_stepdown_families": holm_families,
            "m0_forensics_and_ht_cis": m0_forensics,
        },
        "k3_e4_clean_recoverable_180d_and_r0": {
            "e4_clean_180d": e4_clean_results,
            "r0_rc035_to_rc040": rc_rows,
            "rc036_pre_touch_by_ecosystem": rc036_by_eco,
            "post_20260504_censored_anchors_total": int(anchor_after_20260504.sum()),
            "post_20260504_censored_anchors_recoverable": int((anchor_after_20260504 & is_recov).sum()),
        },
    }

    summary_path = COMPANION_DIR / "measurement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_files = [
        
        "measurement_summary.json",
        "candidate_removals.csv",
        "registry_inflation.csv",
        "event_time_contrasts.csv",
        "advisory_and_bot_contrasts.csv",
        "competing_incidence.csv",
    ]
    manifest = {
        "generated_on": "2026-11-16",
        "files": {fn: sha256_file(COMPANION_DIR / fn) for fn in manifest_files},
    }
    (COMPANION_DIR / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Measurement analysis written to", summary_path)


if __name__ == "__main__":
    main()
