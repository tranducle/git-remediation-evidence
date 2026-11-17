"""Recompute activity-stratified outcomes and sensitivity analyses."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, norm, t as student_t


PACK_ROOT = Path(__file__).resolve().parent.parent
PAPER_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = Path(__file__).resolve().parent.parent
OUT_85 = Path(__file__).resolve().parent.parent / "data"
I0_DIR = Path(__file__).resolve().parent
PIPE_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


estimators = _load_module("_s8_estimators", I0_DIR / "estimators.py")
signal_models = _load_module("_s8_signal_models", PIPE_DIR / "signal_models.py")
version_rules = _load_module("_s8_version_rules", Path(__file__).resolve().parent / "version_rules.py")


def extract_owner(repo_url: str) -> str:
    s = str(repo_url or "").strip().rstrip("/")
    parts = s.split("/")
    return parts[-2].lower() if len(parts) >= 2 else s.lower()


def cluster_ci(est: float, phi: np.ndarray, ca: np.ndarray, cb: np.ndarray, level: float = 0.95) -> dict[str, Any]:
    n_a = int(len(np.unique(ca)))
    n_b = int(len(np.unique(cb)))
    if len(phi) < 2 or n_a < 2 or n_b < 2:
        return {"estimable": False, "estimate": float(est), "n": int(len(phi)), "n_cluster_a": n_a, "n_cluster_b": n_b}
    var = estimators.cluster_variance(phi, ca, cb, adjust=True)
    if not math.isfinite(var) or var <= 0.0:
        return {"estimable": False, "estimate": float(est), "n": int(len(phi)), "n_cluster_a": n_a, "n_cluster_b": n_b}
    df = max(1, min(n_a, n_b) - 1)
    se = math.sqrt(var)
    alpha = 1.0 - level
    crit = float(student_t.ppf(1.0 - alpha / 2.0, df))
    p_val = float(2.0 * student_t.sf(abs(est / se), df))
    return {
        "estimable": True,
        "n": int(len(phi)),
        "n_cluster_a": n_a,
        "n_cluster_b": n_b,
        "df": df,
        "estimate": float(est),
        "se": float(se),
        "lower": float(est - crit * se),
        "upper": float(est + crit * se),
        "p_value": p_val,
    }


def tost_against_bound(est: float, se: float, df: int, bound: float) -> dict[str, float]:
    t_low = (est - (-bound)) / se
    t_high = (bound - est) / se
    p_low = float(student_t.sf(t_low, df))
    p_high = float(student_t.sf(t_high, df))
    crit90 = float(student_t.ppf(0.95, df))
    crit9875 = float(student_t.ppf(1.0 - 0.0125 / 2.0, df))
    return {
        "bound": bound,
        "p_above_neg_bound": p_low,
        "p_below_pos_bound": p_high,
        "tost_p_value": max(p_low, p_high),
        "ci90_lower": float(est - crit90 * se),
        "ci90_upper": float(est + crit90 * se),
        "ci9875_lower": float(est - crit9875 * se),
        "ci9875_upper": float(est + crit9875 * se),
    }


def mean_ci(vals: np.ndarray, mask: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool) & np.isfinite(vals)
    if int(m.sum()) < 2:
        return {"estimable": False, "n": int(m.sum())}
    w = np.ones(len(vals), dtype=float)
    est, inf = estimators.weighted_mean_influence(vals, m, w)
    res = cluster_ci(float(est), inf[m], ca[m], cb[m])
    res["sum_positive"] = int(np.sum(vals[m] == 1.0))
    return res


def diff_means_ci(vals: np.ndarray, tr: np.ndarray, ct: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> dict[str, Any]:
    t_m = np.asarray(tr, dtype=bool) & np.isfinite(vals)
    c_m = np.asarray(ct, dtype=bool) & np.isfinite(vals)
    if int(t_m.sum()) < 2 or int(c_m.sum()) < 2:
        return {"estimable": False, "n_treated": int(t_m.sum()), "n_control": int(c_m.sum())}
    w = np.ones(len(vals), dtype=float)
    mt, it = estimators.weighted_mean_influence(vals, t_m, w)
    mc, ic = estimators.weighted_mean_influence(vals, c_m, w)
    used = t_m | c_m
    res = cluster_ci(float(mt - mc), (it - ic)[used], ca[used], cb[used])
    res["treated_mean"] = float(mt)
    res["control_mean"] = float(mc)
    res["n_treated"] = int(t_m.sum())
    res["n_control"] = int(c_m.sum())
    return res


def korn_graubard_ci(p_hat: float, n_eff: float, alpha: float = 0.05) -> tuple[float, float]:
    x_eff = p_hat * n_eff
    if x_eff <= 0.0:
        lo = 0.0
    else:
        lo = float(beta_dist.ppf(alpha / 2.0, x_eff, n_eff - x_eff + 1.0))
    if x_eff >= n_eff:
        hi = 1.0
    else:
        hi = float(beta_dist.ppf(1.0 - alpha / 2.0, x_eff + 1.0, n_eff - x_eff))
    return lo, hi


def holm_adjust(p_map: dict[str, float]) -> dict[str, float]:
    m = len(p_map)
    finite = sorted(p_map.items(), key=lambda kv: kv[1])
    out: dict[str, float] = {}
    running = 0.0
    for idx, (k, pv) in enumerate(finite):
        adj = min(1.0, pv * (m - idx))
        running = max(running, adj)
        out[k] = running
    return out


def manski_two_arm(y: np.ndarray, r: np.ndarray, tr: np.ndarray, ct: np.ndarray) -> dict[str, Any]:
    t = np.asarray(tr, dtype=bool)
    c = np.asarray(ct, dtype=bool)
    obs = np.asarray(r, dtype=bool)
    nt, nt_o = int(t.sum()), int((t & obs).sum())
    nc, nc_o = int(c.sum()), int((c & obs).sum())
    pt = nt_o / nt if nt else 0.0
    pc = nc_o / nc if nc else 0.0
    mut = float(np.mean(y[t & obs])) if nt_o else 0.0
    muc = float(np.mean(y[c & obs])) if nc_o else 0.0
    t_lo, t_hi = pt * mut, pt * mut + (1.0 - pt)
    c_lo, c_hi = pc * muc, pc * muc + (1.0 - pc)
    return {
        "n_total_frame": nt + nc,
        "n_observed_frame": nt_o + nc_o,
        "unknown_rate_frame": float(1.0 - (nt_o + nc_o) / (nt + nc)) if (nt + nc) else 0.0,
        "n_treated_total": nt,
        "n_treated_observed": nt_o,
        "treated_recoverability": float(pt),
        "treated_obs_mean": mut,
        "n_control_total": nc,
        "n_control_observed": nc_o,
        "control_recoverability": float(pc),
        "control_obs_mean": muc,
        "complete_case_diff": float(mut - muc),
        "manski_lower": float(t_lo - c_hi),
        "manski_upper": float(t_hi - c_lo),
    }


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results/recomputed/activity_analysis.json"


def main() -> None:
    git_obs = pd.read_parquet(OUT_85 / "git_observations.parquet")
    first_exit = pd.read_parquet(OUT_85 / "first_exit.parquet")
    covs = pd.read_parquet(OUT_85 / "covariates.parquet")
    a0_df = pd.DataFrame(json.loads((OUT_85 / "measurement_variants.json").read_text(encoding="utf-8")))
    r0_df = pd.DataFrame(json.loads((OUT_85 / "robustness_panel.json").read_text(encoding="utf-8")))

    e1_npz = dict(np.load(OUT_85 / "signal_panels" / "epss_panel.npz", allow_pickle=False))
    e2_npz = dict(np.load(OUT_85 / "signal_panels" / "advisory_panel.npz", allow_pickle=False))
    e3_npz = dict(np.load(OUT_85 / "signal_panels" / "kev_panel.npz", allow_pickle=False))
    e4_npz = dict(np.load(OUT_85 / "signal_panels" / "competing_events.npz", allow_pickle=False))

    n_total = len(git_obs)
    repo_arr = git_obs["repository_id"].astype(str).to_numpy()
    owner_arr = np.array([extract_owner(u) for u in repo_arr], dtype=object)
    cve_arr = git_obs["cve_id"].astype(str).to_numpy()
    eco_arr = git_obs["ecosystem"].astype(str).str.lower().to_numpy()
    eco_cased = git_obs["ecosystem"].astype(str).to_numpy()
    obs_status = git_obs["observation_status"].astype(str).to_numpy()
    is_recov = obs_status != "unknown"
    is_strict_obs = obs_status == "observed"
    state_arr = np.array([-1 if pd.isna(x) else int(x) for x in git_obs["state"]], dtype=int)
    elig_only = np.array([bool(x) if pd.notna(x) else False for x in git_obs["eligibility_only"]], dtype=bool)
    dep_scope = git_obs["dependency_scope"].astype(str).to_numpy()
    is_prod_scope = dep_scope == "default"

    lag_commits = covs["lagged_commits_90d"].fillna(0.0).to_numpy(dtype=float)
    is_active_90d = lag_commits > 0.0
    is_archived = np.array([bool(x) if pd.notna(x) else False for x in r0_df["is_archived_repo"]], dtype=bool)
    pre_bot_cov = np.array([bool(x) if pd.notna(x) else False for x in covs["pre_signal_bot_present"]], dtype=bool)
    y_pool_13w = np.array([1.0 if (bool(x) if pd.notna(x) else False) else 0.0 for x in covs["outcome_pooled_13w"]], dtype=float)
                                                                                                
    y_act_arr = a0_df["git_history_verified_removal"].to_numpy(dtype=float)
    y_act_plus_s3 = np.where((y_act_arr == 1.0) | (state_arr == 3), 1.0, 0.0)
    p_ln_arr = a0_df["legacy_release_proxy_naive_removal"].to_numpy(dtype=float)
    sig_exposed = a0_df["signal_exposed"].to_numpy(dtype=int) == 1

                                                                                        
    tid_to_idx = {str(git_obs.loc[i, "anchor_id"]): i for i in range(n_total)}
    anchor_after_20260504 = np.zeros(n_total, dtype=bool)
    unreachable_clone = np.zeros(n_total, dtype=bool)
    with (OUT_85 / "exposure_metadata.jsonl").open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            bmr = str(rec.get("baseline_missing_reason") or "none")
            if bmr != "none":
                unreachable_clone[idx] = True
            sig_utc = str(rec.get("signal_time_utc") or "")
            if sig_utc[:10] > "2026-05-04":
                anchor_after_20260504[idx] = True

    exact_pinned = np.zeros(n_total, dtype=bool)
    seen_ws: set[str] = set()
    with (OUT_85 / "baseline_constraints.csv").open(encoding="utf-8", newline="") as f_ws:
        for wrow in csv.DictReader(f_ws):
            tid_w = str(wrow.get("anchor_id") or "")
            if tid_w in seen_ws:
                continue
            seen_ws.add(tid_w)
            idx_w = tid_to_idx.get(tid_w, -1)
            if idx_w < 0 or not is_recov[idx_w]:
                continue
            mev_raw = str(wrow.get("manifest_evidence") or "").strip()
            if mev_raw:
                try:
                    c_str = str(json.loads(mev_raw).get("constraint") or "").strip()
                    if c_str and version_rules.exact_pin(c_str, eco_cased[idx_w]) is not None:
                        exact_pinned[idx_w] = True
                except Exception:
                    pass

                                                                                                
    e1_y = np.asarray(e1_npz["y"], dtype=float)
    e1_coh = np.asarray(e1_npz["cohort"], dtype=float)
    e1_org_coh = np.asarray(e1_npz["organic_cohort"], dtype=float)
    e1_obs = np.asarray(e1_npz["observed"], dtype=bool)
    e1_zero_poc = ~np.asarray(e1_npz["pre_poc_present"], dtype=bool)
    e1_elig = np.asarray(e1_npz["eligibility"], dtype=bool) & e1_zero_poc[:, None]
    e1_shock = np.asarray(e1_npz["calendar_shock_id"]).astype(str)
    is_org_row = np.isfinite(e1_org_coh) | (e1_shock == "organic_threshold_crossing")
    elig_pure = e1_elig & (~is_org_row)[:, None]
    coh_pure = np.where(is_org_row, np.inf, e1_coh)

    e2_y = np.asarray(e2_npz["y"], dtype=float)
    e2_coh = np.asarray(e2_npz["cohort"], dtype=float)
    e2_obs = np.asarray(e2_npz["observed"], dtype=bool)
    e2_elig = np.asarray(e2_npz["eligibility"], dtype=bool)

    fe_ev_t = pd.to_numeric(first_exit["first_event_time"], errors="coerce").to_numpy(dtype=float)
    fe_ce_t = pd.to_numeric(first_exit["censor_time"], errors="coerce").to_numpy(dtype=float)
    fe_cause = pd.to_numeric(first_exit["first_cause"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    real_time = np.clip(
        np.where(np.isfinite(fe_ev_t) & (fe_ev_t > 0.0), fe_ev_t, np.where(np.isfinite(fe_ce_t) & (fe_ce_t > 0.0), fe_ce_t, 180.0)),
        1.0,
        180.0,
    )
    real_status = np.zeros(n_total, dtype=int)
    real_status[np.isin(fe_cause, [1, 2])] = 1
    real_status[fe_cause == 3] = 2
    real_status[fe_cause == 4] = 3

    oor_arr = np.asarray(e4_npz["out_of_range"], dtype=float) == 1.0
    maj_arr = np.asarray(e4_npz["major_change"], dtype=float) == 1.0
    e4_sig = np.asarray(e4_npz["ghsa_reviewed"], dtype=float)

    cand_reasons = {
        "direct removal candidate requires successor/retained-use audit",
        "new coordinates require successor review",
    }
    cand_mask = first_exit["censor_reason"].isin(cand_reasons).to_numpy()

    def run_did_sub(y_mat: np.ndarray, coh_v: np.ndarray, elig_m: np.ndarray, obs_m: np.ndarray, row_mask: np.ndarray) -> dict[str, Any]:
        el = elig_m & row_mask[:, None]
        try:
            agg = estimators.did_aggregate(y_mat, coh_v, 13, anticipation=0, eligibility=el, observed=obs_m)
            u = agg["used"]
            ci = cluster_ci(float(agg["estimate"]), agg["influence"][u], repo_arr[u], cve_arr[u])
            ci["n_used_dyads"] = int(u.sum())
            return ci
        except Exception as exc:
            return {"estimable": False, "reason": str(exc)}

    def run_aj_sub(mask_sub: np.ndarray, cause_idx: int, status_vec: np.ndarray) -> dict[str, Any]:
        m1 = mask_sub & (e4_sig == 1.0)
        m0 = mask_sub & (e4_sig == 0.0)
        n_ev = int(np.sum((status_vec[mask_sub] == (cause_idx + 1)) & (real_time[mask_sub] <= 180.0)))
        if int(m1.sum()) < 2 or int(m0.sum()) < 2 or n_ev == 0:
            return {"estimable": False, "n_obs": int(mask_sub.sum()), "n_treated": int(m1.sum()), "n_control": int(m0.sum()), "n_events": n_ev}
        fit1 = estimators.aj_cif(real_time[m1], status_vec[m1], 180.0, causes=3)
        fit0 = estimators.aj_cif(real_time[m0], status_vec[m0], 180.0, causes=3)
        diff = float(fit1["cif"][cause_idx] - fit0["cif"][cause_idx])
        phi = np.zeros(n_total, dtype=float)
        phi[m1] = fit1["influence"][:, cause_idx]
        phi[m0] = -fit0["influence"][:, cause_idx]
        ci = cluster_ci(diff, phi[mask_sub], repo_arr[mask_sub], cve_arr[mask_sub])
        ci.update({
            "n_obs": int(mask_sub.sum()),
            "n_treated": int(m1.sum()),
            "n_control": int(m0.sum()),
            "n_events": n_ev,
            "cif_treated": float(fit1["cif"][cause_idx]),
            "cif_control": float(fit0["cif"][cause_idx]),
        })
        return ci

    f002_strata = {}
    for s_name, s_mask in [
        ("active_90d_recoverable", is_recov & is_active_90d),
        ("dormant_90d_recoverable", is_recov & (~is_active_90d)),
        ("prod_default_scope_recoverable", is_recov & is_prod_scope),
        ("nonprod_dev_test_scope_recoverable", is_recov & (~is_prod_scope)),
        ("active_90d_and_prod_scope_recoverable", is_recov & is_active_90d & is_prod_scope),
        ("non_archived_recoverable", is_recov & (~is_archived)),
    ]:
        row_filter = s_mask
                                                                                                                              
        cov_only = {
            "active_90d_recoverable": is_active_90d,
            "dormant_90d_recoverable": ~is_active_90d,
            "prod_default_scope_recoverable": is_prod_scope,
            "nonprod_dev_test_scope_recoverable": ~is_prod_scope,
            "active_90d_and_prod_scope_recoverable": is_active_90d & is_prod_scope,
            "non_archived_recoverable": ~is_archived,
        }[s_name]
        ub_st = real_status.copy()
        ub_st[cand_mask] = 3
        f002_strata[s_name] = {
            "n_recoverable": int(row_filter.sum()),
            "share_of_recoverable": float(row_filter.sum() / is_recov.sum()),
            "active_13w_rate": float(np.mean(y_act_arr[row_filter])),
            "active_13w_count": int(np.sum(y_act_arr[row_filter] == 1.0)),
            "proxy_13w_rate": float(np.mean(p_ln_arr[row_filter])),
            "k4_inflation_ci": mean_ci(p_ln_arr - y_act_arr, row_filter, repo_arr, cve_arr),
            "e1_pooled_att": run_did_sub(e1_y, e1_coh, e1_elig, e1_obs, cov_only),
            "e1_pure_switch_att": run_did_sub(e1_y, coh_pure, elig_pure, e1_obs, cov_only),
            "e2_ghsa_att": run_did_sub(e2_y, e2_coh, e2_elig, e2_obs, cov_only),
            "e4_upgrade_180d": run_aj_sub(row_filter, 0, real_status),
            "e4_candidate_removal_ub_180d": run_aj_sub(row_filter | (cand_mask & cov_only), 2, ub_st),
            "pre_bot_n": int((row_filter & pre_bot_cov).sum()),
            "pre_bot_active_rate": float(np.mean(y_act_arr[row_filter & pre_bot_cov])) if int((row_filter & pre_bot_cov).sum()) else None,
        }

                                                                                           
                                       
                                                               
    overlap_df = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "observability" / "overlap.csv")
    cov_flow_df = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "observability" / "coverage_flow.csv")
                                                             
    pre_bot_active_diff = diff_means_ci(y_act_arr, is_recov & pre_bot_cov, is_recov & (~pre_bot_cov), repo_arr, cve_arr)
    pre_bot_plus_s3_diff = diff_means_ci(y_act_plus_s3, is_recov & pre_bot_cov, is_recov & (~pre_bot_cov), repo_arr, cve_arr)
    pre_bot_nonaction_ci = mean_ci(1.0 - y_act_arr, is_recov & pre_bot_cov, repo_arr, cve_arr)

                                                                 
    ps_e2 = np.asarray(e2_npz["bot_propensity"], dtype=float)
    bot_e2 = np.asarray(e2_npz["pre_signal_bot"], dtype=float)
    sup_bot = np.isfinite(bot_e2) & np.isfinite(ps_e2) & (ps_e2 >= 0.05) & (ps_e2 <= 0.95)
    agg_e2_main = estimators.did_aggregate(e2_y, e2_coh, 13, anticipation=0, eligibility=e2_elig, observed=e2_obs)
    agg_bot1 = estimators.did_aggregate(e2_y, e2_coh, 13, anticipation=0, eligibility=e2_elig & (sup_bot & (bot_e2 == 1.0))[:, None], observed=e2_obs)
    agg_bot0 = estimators.did_aggregate(e2_y, e2_coh, 13, anticipation=0, eligibility=e2_elig & (sup_bot & (bot_e2 == 0.0))[:, None], observed=e2_obs)
    agg_e2_sup_all = estimators.did_aggregate(e2_y, e2_coh, 13, anticipation=0, eligibility=e2_elig & sup_bot[:, None], observed=e2_obs)

                                                                                          
    e2_tr_rec = np.isfinite(e2_coh) & is_recov
    e2_ct_rec = np.isinf(e2_coh) & is_recov
    e2_s3_incl_main = diff_means_ci(y_act_plus_s3, e2_tr_rec, e2_ct_rec, repo_arr, cve_arr)
    e2_s3_incl_bot1 = diff_means_ci(y_act_plus_s3, e2_tr_rec & pre_bot_cov, e2_ct_rec & pre_bot_cov, repo_arr, cve_arr)
    e2_s3_incl_bot0 = diff_means_ci(y_act_plus_s3, e2_tr_rec & (~pre_bot_cov), e2_ct_rec & (~pre_bot_cov), repo_arr, cve_arr)

                                                          
    source_915 = {}
    for cand_bool_name, cand_b in [
        ("is_recov & pre_bot_cov", is_recov & pre_bot_cov),
        ("any_obs_e2 & pre_bot_cov", np.any(e2_obs, axis=1) & pre_bot_cov),
        ("matched_bot_paths_nonempty_recov", is_recov & covs["matched_bot_paths"].fillna("").astype(str).ne("").to_numpy()),
        ("r0_config_activated_recov", is_recov & (r0_df["config_activated"].fillna(0).to_numpy(dtype=float) > 0)),
        ("any_obs_e2 & config_activated", np.any(e2_obs, axis=1) & (r0_df["config_activated"].fillna(0).to_numpy(dtype=float) > 0)),
    ]:
        n_b = int(cand_b.sum())
        n_inact = int(np.sum(cand_b & (y_act_arr == 0.0)))
        source_915[cand_bool_name] = {"n": n_b, "inactive": n_inact}

                                                                     
    m0_bundle = json.loads((OUT_85 / "workflow" / "sample.json").read_text(encoding="utf-8"))
    pkts_root = OUT_85 / "workflow" / "records"
    m0_cases = {str(c["case_id"]): c for c in m0_bundle["sampled_cases"]}
    m0_records = []
    for pdir in sorted(pkts_root.iterdir()):
        if not pdir.is_dir():
            continue
        tr = json.loads((pdir / "trace_record.json").read_text(encoding="utf-8"))
        cid = str(tr["case_id"])
        sc = m0_cases[cid]
        tid = str(sc.get("anchor_id") or "")
        r_idx = tid_to_idx.get(tid, -1)
        recov_flag = bool(r_idx >= 0 and is_recov[r_idx])
        pr_txt = str(tr.get("pr_timeline_text") or "")
        bot_flag = ("bot_config=present" in pr_txt) or bool(r_idx >= 0 and pre_bot_cov[r_idx])
        w_ht = float(sc.get("sampling_weight") or (1.0 / float(sc.get("inclusion_probability", 1.0))))
        if not recov_flag:
            cat4 = "non_identifiable_unknown_git"
        elif tr.get("outcome_unchanged"):
            cat4 = "recoverable_verified_unchanged"
        elif tr.get("has_public_pr_signal"):
            cat4 = "recoverable_advisory_precedes_pr_and_merge"
        else:
            cat4 = "recoverable_manual_commit_no_pr"
        m0_records.append({
            "case_id": cid,
            "cat4": cat4,
            "recov": recov_flag,
            "bot_present": bot_flag,
            "unchanged": bool(tr.get("outcome_unchanged")),
            "signal_type": str(tr.get("signal_type") or ""),
            "weight": w_ht,
        })
    m0_df = pd.DataFrame(m0_records)
    w_all = m0_df["weight"].to_numpy(dtype=float)
    n_eff_all = float((w_all.sum() ** 2) / np.sum(w_all ** 2))
    m0_cat4_ht = {}
    for c4 in [
        "non_identifiable_unknown_git",
        "recoverable_verified_unchanged",
        "recoverable_manual_commit_no_pr",
        "recoverable_advisory_precedes_pr_and_merge",
    ]:
        ind = (m0_df["cat4"].to_numpy() == c4).astype(float)
        p_hat = float(np.dot(w_all, ind) / w_all.sum())
        kg_lo, kg_hi = korn_graubard_ci(p_hat, n_eff_all)
        m0_cat4_ht[c4] = {
            "count_n120": int(ind.sum()),
            "ht_share": p_hat,
            "kish_n_eff": n_eff_all,
            "kg_ci95_lower": kg_lo,
            "kg_ci95_upper": kg_hi,
        }
                                                
    m0_rec_df = m0_df[m0_df["recov"]]
    w_rec = m0_rec_df["weight"].to_numpy(dtype=float)
    n_eff_rec = float((w_rec.sum() ** 2) / np.sum(w_rec ** 2))
    m0_rec_domain = {}
    for c4 in [
        "recoverable_verified_unchanged",
        "recoverable_manual_commit_no_pr",
        "recoverable_advisory_precedes_pr_and_merge",
    ]:
        ind = (m0_rec_df["cat4"].to_numpy() == c4).astype(float)
        p_hat = float(np.dot(w_rec, ind) / w_rec.sum())
        kg_lo, kg_hi = korn_graubard_ci(p_hat, n_eff_rec)
        m0_rec_domain[c4] = {
            "count_n25": int(ind.sum()),
            "domain_ht_share": p_hat,
            "kish_n_eff_domain": n_eff_rec,
            "kg_ci95_lower": kg_lo,
            "kg_ci95_upper": kg_hi,
        }

                                                                                                    
    agg_e1_pool = estimators.did_aggregate(e1_y, e1_coh, 13, anticipation=0, eligibility=e1_elig, observed=e1_obs)
    u_e1 = agg_e1_pool["used"]
    ci_e1_pool = cluster_ci(float(agg_e1_pool["estimate"]), agg_e1_pool["influence"][u_e1], repo_arr[u_e1], cve_arr[u_e1])
    tost_e1_pool = tost_against_bound(ci_e1_pool["estimate"], ci_e1_pool["se"], ci_e1_pool["df"], 0.05)

    agg_e1_pure = estimators.did_aggregate(e1_y, coh_pure, 13, anticipation=0, eligibility=elig_pure, observed=e1_obs)
    u_pure = agg_e1_pure["used"]
    ci_e1_pure = cluster_ci(float(agg_e1_pure["estimate"]), agg_e1_pure["influence"][u_pure], repo_arr[u_pure], cve_arr[u_pure])
    tost_e1_pure = tost_against_bound(ci_e1_pure["estimate"], ci_e1_pure["se"], ci_e1_pure["df"], 0.05)

                                                        
    rq1_csv = pd.read_csv(Path(__file__).resolve().parent.parent / "results" / "signals" / "epss_effects.csv")
    rq1_rows = rq1_csv[
        ["estimand", "subgroup", "estimate", "se", "lower", "upper", "p_value", "n_fix_dyads", "n_calendar_shocks", "uncertainty_method", "test_status", "failure_reason"]
    ].to_dict(orient="records")

    ci_e2_main = cluster_ci(float(agg_e2_main["estimate"]), agg_e2_main["influence"][agg_e2_main["used"]], repo_arr[agg_e2_main["used"]], cve_arr[agg_e2_main["used"]])
    tost_e2_main = tost_against_bound(ci_e2_main["estimate"], ci_e2_main["se"], ci_e2_main["df"], 0.04)

                                                                                                              
    a0_paired_diff_cis = {}
    for eco_l in ["pooled", "npm", "pypi", "maven", "composite_74673"]:
        if eco_l == "composite_74673":
            m = np.ones(n_total, dtype=bool)
        elif eco_l == "pooled":
            m = is_recov
        else:
            m = is_recov & (eco_arr == eco_l)
        tr = m & sig_exposed
        ct = m & (~sig_exposed)
        w = np.ones(n_total, dtype=float)
        mt_p, it_p = estimators.weighted_mean_influence(p_ln_arr, tr, w)
        mc_p, ic_p = estimators.weighted_mean_influence(p_ln_arr, ct, w)
        mt_g, it_g = estimators.weighted_mean_influence(y_act_arr, tr, w)
        mc_g, ic_g = estimators.weighted_mean_influence(y_act_arr, ct, w)
        att_p = float(mt_p - mc_p)
        att_g = float(mt_g - mc_g)
        diff_pg = float(att_p - att_g)
        phi_pg = ((it_p - ic_p) - (it_g - ic_g))[m]
        ci_pg = cluster_ci(diff_pg, phi_pg, repo_arr[m], cve_arr[m])
        ci_p = cluster_ci(att_p, (it_p - ic_p)[m], repo_arr[m], cve_arr[m])
        ci_g = cluster_ci(att_g, (it_g - ic_g)[m], repo_arr[m], cve_arr[m])
        a0_paired_diff_cis[eco_l] = {
            "n": int(m.sum()),
            "proxy_att": att_p,
            "proxy_att_ci": {k: ci_p.get(k) for k in ("lower", "upper", "p_value")},
            "git_att": att_g,
            "git_att_ci": {k: ci_g.get(k) for k in ("lower", "upper", "p_value")},
            "diff_proxy_minus_git": diff_pg,
            "diff_proxy_minus_git_ci": {k: ci_pg.get(k) for k in ("se", "lower", "upper", "p_value")},
        }

                                                                                                                             
    ub_status_all = real_status.copy()
    ub_status_all[cand_mask] = 3
    e4_ub_res = run_aj_sub(is_recov | cand_mask, 2, ub_status_all)
    p_e4_ub = float(e4_ub_res["p_value"])
    p_e4_ver = 1.0
    p_e3_perm = 0.3889
    p_a0_clean = float(a0_paired_diff_cis["pooled"]["diff_proxy_minus_git_ci"]["p_value"])
    p_bot_inter = 0.37111127

    clean_holm_primary = {
        "H4_clean_verified_removal": holm_adjust({"E1": ci_e1_pool["p_value"], "E2": ci_e2_main["p_value"], "E3": p_e3_perm, "E4": p_e4_ver}),
        "H4_clean_candidate_ub": holm_adjust({"E1": ci_e1_pool["p_value"], "E2": ci_e2_main["p_value"], "E3": p_e3_perm, "E4": p_e4_ub}),
        "H4_pure_switch_candidate_ub": holm_adjust({"E1": ci_e1_pure["p_value"], "E2": ci_e2_main["p_value"], "E3": p_e3_perm, "E4": p_e4_ub}),
        "H5a_clean_with_paired_A0": holm_adjust({"E1": ci_e1_pool["p_value"], "E2": ci_e2_main["p_value"], "E3": p_e3_perm, "E4": p_e4_ub, "A0": p_a0_clean}),
        "H5b_clean_with_Bot": holm_adjust({"E1": ci_e1_pool["p_value"], "E2": ci_e2_main["p_value"], "E3": p_e3_perm, "E4": p_e4_ub, "Bot": p_bot_inter}),
        "H6_clean_with_paired_A0_and_Bot": holm_adjust({"E1": ci_e1_pool["p_value"], "E2": ci_e2_main["p_value"], "E3": p_e3_perm, "E4": p_e4_ub, "A0": p_a0_clean, "Bot": p_bot_inter}),
    }

                                                           
    y_fut_30 = r0_df["y_placebo_future_30d"].to_numpy(dtype=float)
    rc036_decomp = {}
    for scope_lbl, s_m in [("all_74673", np.ones(n_total, dtype=bool)), ("recoverable_13785", is_recov)]:
        rc036_decomp[scope_lbl] = {}
        for eco_l in ["pooled", "npm", "pypi", "maven"]:
            m = s_m if eco_l == "pooled" else (s_m & (eco_arr == eco_l))
            rc036_decomp[scope_lbl][eco_l] = diff_means_ci(y_fut_30, m & sig_exposed, m & (~sig_exposed), repo_arr, cve_arr)

                                                                  
    se_act = 0.6725
    se_s2 = 0.3986
    recall_sensitivity = {
        "e1_pooled_div_06725": {"est": ci_e1_pool["estimate"] / se_act, "lower": ci_e1_pool["lower"] / se_act, "upper": ci_e1_pool["upper"] / se_act},
        "e1_pure_switch_div_06725": {"est": ci_e1_pure["estimate"] / se_act, "lower": ci_e1_pure["lower"] / se_act, "upper": ci_e1_pure["upper"] / se_act},
        "e2_ghsa_unweighted_div_06725": {"est": ci_e2_main["estimate"] / se_act, "lower": ci_e2_main["lower"] / se_act, "upper": ci_e2_main["upper"] / se_act},
        "e2_ghsa_ipw_div_06725": {"est": 0.03128434 / se_act, "lower": 0.00250977 / se_act, "upper": 0.06005892 / se_act},
    }

                                                                                                
    pre_may4 = ~anchor_after_20260504
    true_semver_oor = oor_arr & (~exact_pinned)
    f006_k3 = {
        "b1_composition_recoverable": {
            "n_oor_total": int((is_recov & oor_arr).sum()),
            "n_exact_pinned_in_oor": int((is_recov & oor_arr & exact_pinned).sum()),
            "n_true_semver_oor_excluding_exact_pins": int((is_recov & true_semver_oor).sum()),
            "n_major_change_in_recov": int((is_recov & maj_arr).sum()),
            "n_complete_180d_pre_20260504_recov": int((is_recov & pre_may4).sum()),
            "n_truncated_post_20260504_recov": int((is_recov & anchor_after_20260504).sum()),
            "n_49662_unique_fix_dyad_ids_in_74673": int(len(np.unique(np.asarray(e4_npz["fix_dyad_id"])))),
            "n_e4_treated_total_74673": int(np.sum(e4_sig == 1.0)),
            "n_e4_control_total_74673": int(np.sum(e4_sig == 0.0)),
        },
        "upgrade_180d_exact_pinned_recov": run_aj_sub(is_recov & oor_arr & exact_pinned, 0, real_status),
        "upgrade_180d_true_semver_oor_recov": run_aj_sub(is_recov & true_semver_oor, 0, real_status),
        "upgrade_180d_major_change_recov": run_aj_sub(is_recov & maj_arr, 0, real_status),
        "upgrade_180d_nonmajor_recov": run_aj_sub(is_recov & (~maj_arr), 0, real_status),
        "complete_180d_pre20260504": {
            "pooled_upgrade": run_aj_sub(is_recov & pre_may4, 0, real_status),
            "in_range_upgrade": run_aj_sub(is_recov & pre_may4 & (~oor_arr), 0, real_status),
            "out_of_range_upgrade": run_aj_sub(is_recov & pre_may4 & oor_arr, 0, real_status),
            "exact_pinned_upgrade": run_aj_sub(is_recov & pre_may4 & oor_arr & exact_pinned, 0, real_status),
            "true_semver_oor_upgrade": run_aj_sub(is_recov & pre_may4 & true_semver_oor, 0, real_status),
            "candidate_removal_ub_pooled": run_aj_sub((is_recov | cand_mask) & pre_may4, 2, ub_status_all),
        },
    }

                                                                                            
    def aj_two_group_diff(m_a: np.ndarray, m_b: np.ndarray, cause_idx: int, st_vec: np.ndarray) -> dict[str, Any]:
        fa = estimators.aj_cif(real_time[m_a], st_vec[m_a], 180.0, causes=3)
        fb = estimators.aj_cif(real_time[m_b], st_vec[m_b], 180.0, causes=3)
        d = float(fa["cif"][cause_idx] - fb["cif"][cause_idx])
        phi = np.zeros(n_total, dtype=float)
        phi[m_a] = fa["influence"][:, cause_idx]
        phi[m_b] = -fb["influence"][:, cause_idx]
        u = m_a | m_b
        ci = cluster_ci(d, phi[u], repo_arr[u], cve_arr[u])
        ci["cif_a"] = float(fa["cif"][cause_idx])
        ci["cif_b"] = float(fb["cif"][cause_idx])
        ci["n_a"] = int(m_a.sum())
        ci["n_b"] = int(m_b.sum())
        return ci

                                                    
    def aj_did_oor_x_sig(base_m: np.ndarray, oor_flag: np.ndarray, cause_idx: int, st_vec: np.ndarray) -> dict[str, Any]:
        m_os = base_m & oor_flag & (e4_sig == 1.0)
        m_oc = base_m & oor_flag & (e4_sig == 0.0)
        m_is = base_m & (~oor_flag) & (e4_sig == 1.0)
        m_ic = base_m & (~oor_flag) & (e4_sig == 0.0)
        fos = estimators.aj_cif(real_time[m_os], st_vec[m_os], 180.0, causes=3)
        foc = estimators.aj_cif(real_time[m_oc], st_vec[m_oc], 180.0, causes=3)
        fis = estimators.aj_cif(real_time[m_is], st_vec[m_is], 180.0, causes=3)
        fic = estimators.aj_cif(real_time[m_ic], st_vec[m_ic], 180.0, causes=3)
        did = float((fos["cif"][cause_idx] - foc["cif"][cause_idx]) - (fis["cif"][cause_idx] - fic["cif"][cause_idx]))
        phi = np.zeros(n_total, dtype=float)
        phi[m_os] = fos["influence"][:, cause_idx]
        phi[m_oc] = -foc["influence"][:, cause_idx]
        phi[m_is] = -fis["influence"][:, cause_idx]
        phi[m_ic] = fic["influence"][:, cause_idx]
        u = m_os | m_oc | m_is | m_ic
        return cluster_ci(did, phi[u], repo_arr[u], cve_arr[u])

    f006_k3["post_signal_oor_minus_ir_upgrade"] = aj_two_group_diff(is_recov & oor_arr & (e4_sig == 1.0), is_recov & (~oor_arr) & (e4_sig == 1.0), 0, real_status)
    f006_k3["post_signal_true_semver_oor_minus_ir_upgrade"] = aj_two_group_diff(is_recov & true_semver_oor & (e4_sig == 1.0), is_recov & (~oor_arr) & (e4_sig == 1.0), 0, real_status)
    f006_k3["did_oor_x_signaled_upgrade"] = aj_did_oor_x_sig(is_recov, oor_arr, 0, real_status)
    f006_k3["did_true_semver_oor_x_signaled_upgrade"] = aj_did_oor_x_sig(is_recov, true_semver_oor, 0, real_status)
    f006_k3["post_signal_oor_minus_ir_candidate_removal_ub"] = aj_two_group_diff((is_recov | cand_mask) & oor_arr & (e4_sig == 1.0), (is_recov | cand_mask) & (~oor_arr) & (e4_sig == 1.0), 2, ub_status_all)
    f006_k3["post_signal_major_minus_nonmajor_candidate_removal_ub"] = aj_two_group_diff((is_recov | cand_mask) & maj_arr & (e4_sig == 1.0), (is_recov | cand_mask) & (~maj_arr) & (e4_sig == 1.0), 2, ub_status_all)
    f006_k3["did_oor_x_signaled_candidate_removal_ub"] = aj_did_oor_x_sig(is_recov | cand_mask, oor_arr, 2, ub_status_all)

                                                                                  
    s1_ev = (fe_cause == 1) & (real_time <= 180.0)
    s2_ev = (fe_cause == 2) & (real_time <= 180.0)
    f006_k3["state1_vs_state2_180d_counts"] = {
        "in_range_s1": int(np.sum(is_recov & (~oor_arr) & s1_ev)),
        "in_range_s2": int(np.sum(is_recov & (~oor_arr) & s2_ev)),
        "oor_all_s1": int(np.sum(is_recov & oor_arr & s1_ev)),
        "oor_all_s2": int(np.sum(is_recov & oor_arr & s2_ev)),
        "true_semver_oor_s1": int(np.sum(is_recov & true_semver_oor & s1_ev)),
        "true_semver_oor_s2": int(np.sum(is_recov & true_semver_oor & s2_ev)),
        "state2_recall_inflated_oor_upgrade_diff_approx": float(
            f006_k3["upgrade_180d_true_semver_oor_recov"]["estimate"] / se_s2
        ) if f006_k3["upgrade_180d_true_semver_oor_recov"].get("estimable") else None,
    }

                                                                                                 
    e1_tr_z = np.isfinite(e1_coh) & e1_zero_poc
    e1_ct_z = np.isinf(e1_coh) & e1_zero_poc
    e2_tr = np.isfinite(e2_coh)
    e2_ct = np.isinf(e2_coh)
    e3_coh = np.asarray(e3_npz["cohort"], dtype=float)
    e3_tr = np.isfinite(e3_coh)
    e3_ct = np.isinf(e3_coh)

                                                                                                                         
                                                                                                                                                                       
    reach = ~unreachable_clone
                                                                                                                                                        
    k4_joint_lo = float(
        (np.sum(p_ln_arr[is_recov] - y_act_arr[is_recov]) + np.sum(p_ln_arr[reach & (~is_recov)] - 1.0) + np.sum(unreachable_clone) * (-1.0))
        / n_total
    )
                                                                                                                                      
    k4_joint_hi = float(
        (np.sum(p_ln_arr[is_recov] - y_act_arr[is_recov]) + np.sum(p_ln_arr[reach & (~is_recov)] - 0.0) + np.sum(unreachable_clone) * (+1.0))
        / n_total
    )

    f007_manski_and_s0elig = {
        "canonical_is_recov_13785_manski": {
            "e1_zero_pre_poc_71689": manski_two_arm(y_act_arr, is_recov, e1_tr_z, e1_ct_z),
            "e1_pure_switch_frame": manski_two_arm(y_act_arr, is_recov, np.isfinite(coh_pure) & e1_zero_poc & (~is_org_row), np.isinf(coh_pure) & e1_zero_poc & (~is_org_row)),
            "e2_ghsa_74673": manski_two_arm(y_act_arr, is_recov, e2_tr, e2_ct),
            "e3_kev_74673": manski_two_arm(y_act_arr, is_recov, e3_tr, e3_ct),
            "k4_joint_proxy_missing_on_1839_unreachable": {
                "n_total": int(n_total),
                "n_recoverable": int(is_recov.sum()),
                "n_reachable_unrecoverable": int((reach & (~is_recov)).sum()),
                "n_unreachable_clone": int(unreachable_clone.sum()),
                "joint_manski_lower": k4_joint_lo,
                "joint_manski_upper": k4_joint_hi,
            },
        },
        "why_13951_exceeded_13785": {
            "n_any_panel_week_observed": int(np.any(e2_obs, axis=1).sum()),
            "n_is_recov": int(is_recov.sum()),
            "n_diff_partial_week_censored_to_unknown": int(np.sum(np.any(e2_obs, axis=1) & (~is_recov))),
        },
        "strict_observed_13550_excluding_235_elig_only": {
            "n_strict_observed": int(is_strict_obs.sum()),
            "n_elig_only_excluded": int((is_recov & elig_only).sum()),
            "active_13w_rate": float(np.mean(y_act_arr[is_strict_obs])),
            "k4_inflation_ci": mean_ci(p_ln_arr - y_act_arr, is_strict_obs, repo_arr, cve_arr),
        },
    }

                                                               
    accounting_reconciliations = {
        "e1_dyads_reconciliation": {
            "rq1_effects_csv_first_estimand": str(rq1_csv["estimand"].iloc[0]),
            "rq1_effects_csv_n_fix_dyads_pooled": int(rq1_csv["n_fix_dyads"].iloc[0]),
            "did_aggregate_used_rows_pooled": int(u_e1.sum()),
            "did_aggregate_unique_fix_dyad_ids_pooled": int(len(np.unique(np.asarray(e1_npz["fix_dyad_id"])[u_e1]))),
            "did_aggregate_used_rows_pure_switch": int(u_pure.sum()),
            "did_aggregate_unique_fix_dyad_ids_pure_switch": int(len(np.unique(np.asarray(e1_npz["fix_dyad_id"])[u_pure]))),
            "explanation": "epss_effects.csv reports unique fix_dyad_ids (10,941) across u_e1 (13,403 exposure rows), whereas companion pure_model_switch reported exposure rows (u_pure = 12,984 exposure rows across 10,565 unique fix_dyad_ids); removing 419 organic-crossing exposure rows shrinks 13,403 exposure rows to 12,984 exposure rows (and shrinks 10,941 unique fix_dyad_ids to 10,565).",
        },
        "e2_bot_dyads_reconciliation": {
            "pooled_e2_used_rows": int(agg_e2_main["used"].sum()),
            "pooled_e2_unique_fix_dyad_ids": int(len(np.unique(np.asarray(e2_npz["fix_dyad_id"])[agg_e2_main["used"]]))),
            "sup_bot_all_unique_fix_dyad_ids": int(len(np.unique(np.asarray(e2_npz["fix_dyad_id"])[agg_e2_sup_all["used"]]))),
            "bot0_used_rows": int(agg_bot0["used"].sum()),
            "bot0_unique_fix_dyad_ids": int(len(np.unique(np.asarray(e2_npz["fix_dyad_id"])[agg_bot0["used"]]))),
            "bot1_used_rows": int(agg_bot1["used"].sum()),
            "bot1_unique_fix_dyad_ids": int(len(np.unique(np.asarray(e2_npz["fix_dyad_id"])[agg_bot1["used"]]))),
            "shared_fix_dyad_ids_across_bot0_and_bot1": int(
                len(
                    set(np.unique(np.asarray(e2_npz["fix_dyad_id"])[agg_bot0["used"]])).intersection(
                        set(np.unique(np.asarray(e2_npz["fix_dyad_id"])[agg_bot1["used"]]))
                    )
                )
            ),
        },
        "pre_bot_915_vs_861_candidates": source_915,
        "m0_bot_120_partition": {
            "total_120_bot_present": int(m0_df["bot_present"].sum()),
            "total_120_bot_absent": int((~m0_df["bot_present"]).sum()),
            "unchanged_117_bot_present": int(m0_df.loc[m0_df["unchanged"], "bot_present"].sum()),
            "unchanged_117_bot_absent": int((~m0_df.loc[m0_df["unchanged"], "bot_present"]).sum()),
            "changed_3_bot_present": int(m0_df.loc[~m0_df["unchanged"], "bot_present"].sum()),
            "changed_3_bot_absent": int((~m0_df.loc[~m0_df["unchanged"], "bot_present"]).sum()),
        },
    }

    out = {
        "schema_version": "remediation_activity_summary.v1",
        "generated_on": "2026-11-16",
        "f002_activity_and_scope_stratification": f002_strata,
        "f003_bot_baseline_state3_and_m0": {
            "pre_bot_two_arm_active_diff": pre_bot_active_diff,
            "pre_bot_two_arm_state3_inclusive_diff": pre_bot_plus_s3_diff,
            "pre_bot_nonaction_clustered_ci": pre_bot_nonaction_ci,
            "e2_state3_inclusive_13w_contrasts": {
                "main": e2_s3_incl_main,
                "bot_present": e2_s3_incl_bot1,
                "bot_absent": e2_s3_incl_bot0,
            },
            "m0_four_category_ht_and_korn_graubard": m0_cat4_ht,
            "m0_recoverable_domain_n25_ht_and_korn_graubard": m0_rec_domain,
        },
        "f004_clean_holm_and_tost": {
            "clean_holm_stepdown_families": clean_holm_primary,
            "e1_pooled_tost_5pp": tost_e1_pool,
            "e1_pure_switch_tost_5pp": tost_e1_pure,
            "e2_ghsa_tost_4pp": tost_e2_main,
        },
        "f005_rq1_benchmarks_recall_and_rc036": {
            "rq1_effects_all_rows": rq1_rows,
            "recall_adjusted_bounds": recall_sensitivity,
            "rc036_future_shock_placebo_by_ecosystem": rc036_decomp,
        },
        "f006_k3_disaggregation_and_contrasts": f006_k3,
        "f007_canonical_manski_and_state0_elig": f007_manski_and_s0elig,
        "f010_a0_paired_diff_cis": a0_paired_diff_cis,
        "accounting_reconciliations": accounting_reconciliations,
    }

    destination = OUTPUT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("Activity analysis written to", destination)



if __name__ == "__main__":
    main()
