"""Classify sampled workflow traces with design weights."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

import study_contract
import validation

CONSTRUCT_VERSION_M0 = 'm0-automated-trace-v4.0-rev4'
MAX_M0_SAMPLE_CASES = 120
M0_PERSON_HOUR_CAP = 0.0

M0_TRACE_CODEBOOK: tuple[str, ...] = (
    'advisory_precedes_pr_and_merge',
    'scheduled_bot_maintenance',
    'manual_maintainer_commit_no_pr',
    'unchanged_with_config',
    'discordant_timing',
    'insufficient_public_trace',
)

WORKFLOW_TRACE_AUDIT_COLUMNS: tuple[str, ...] = (
    'mode_label',
    'stratum_scope',
    'signal_type',
    'ecosystem',
    'trace_category',
    'n_cases',
    'weighted_count',
    'weighted_proportion',
    'se',
    'ci95_low',
    'ci95_high',
    'cohen_kappa_r1_r2',
    'weighted_kappa_r1_r2',
    'prior_v1_hours',
    'm0_hours',
    'total_v1_m0_hours',
    'within_240h_budget',
    'within_rev4_zero_human_cap',
    'author_type',
    'human_attestation',
    'human_validated_claims_withdrawn',
    'output_label',
)


def _parse_dt(value: Any) -> dt.datetime | None:
    if value is None or value == '':
        return None
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.timezone.utc) if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    try:
        parsed = dt.datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
    except ValueError:
        return None
    return parsed.astimezone(dt.timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def compute_cohen_kappa(
    labels_a: list[str],
    labels_b: list[str],
    weights: list[float] | None = None,
) -> dict[str, Any]:
    pass
    if len(labels_a) != len(labels_b):
        raise ValueError('Label lists must have identical length')
    if not labels_a:
        return {'kappa': 1.0, 'p_observed': 1.0, 'p_expected': 1.0, 'n': 0}

    w_arr = np.ones(len(labels_a), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    total_w = float(np.sum(w_arr))
    if total_w <= 0.0:
        raise ValueError('Total weight must be positive')
    w_norm = w_arr / total_w

    cats = sorted(set(labels_a) | set(labels_b))
    p_obs = float(sum(w_norm[i] for i in range(len(labels_a)) if labels_a[i] == labels_b[i]))
    p_exp = 0.0
    for c in cats:
        pa = float(sum(w_norm[i] for i, val in enumerate(labels_a) if val == c))
        pb = float(sum(w_norm[i] for i, val in enumerate(labels_b) if val == c))
        p_exp += pa * pb

    if abs(1.0 - p_exp) < 1e-12:
        kappa = 1.0 if abs(p_obs - 1.0) < 1e-12 else 0.0
    else:
        kappa = (p_obs - p_exp) / (1.0 - p_exp)
    return {
        'kappa': round(float(kappa), 6),
        'p_observed': round(p_obs, 6),
        'p_expected': round(p_exp, 6),
        'n': len(labels_a),
    }


def _weighted_proportion_ci(
    indicators: np.ndarray,
    weights: np.ndarray,
    clusters: list[str],
) -> tuple[float, float, float, float]:
    pass
    w_sum = float(np.sum(weights))
    if w_sum <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    p_hat = float(np.dot(weights, indicators) / w_sum)

    cluster_scores: dict[str, float] = {}
    for cid, y_i, w_i in zip(clusters, indicators, weights):
        score = float(w_i) * (float(y_i) - p_hat) / w_sum
        cluster_scores[cid] = cluster_scores.get(cid, 0.0) + score

    g = len(cluster_scores)
    if g < 2:
        return p_hat, 0.0, max(0.0, p_hat), min(1.0, p_hat)

    vals = np.asarray(list(cluster_scores.values()), dtype=float)
    var_est = (g / float(g - 1)) * float(np.sum((vals - np.mean(vals)) ** 2))
    se = math.sqrt(max(0.0, var_est))
    lo = max(0.0, p_hat - 1.96 * se)
    hi = min(1.0, p_hat + 1.96 * se)
    return p_hat, se, lo, hi


def sample_m0_cases(
    population_rows: list[dict[str, Any]],
    *,
    max_cases: int = MAX_M0_SAMPLE_CASES,
    seed: int = study_contract.MASTER_SEED,
) -> list[dict[str, Any]]:
    pass
    if max_cases < 1 or max_cases > MAX_M0_SAMPLE_CASES:
        raise ValueError(f'M0 sample size cap is 1..{MAX_M0_SAMPLE_CASES}, got {max_cases}')
    if not population_rows:
        return []

    eff_seed = study_contract.derive_child_seed(study_contract.STUDY_ID, 'M0', 'stratified_sample_rev4', int(seed))
    rng = random.Random(eff_seed)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in population_rows:
        sig = str(row.get('signal_type') or 'unknown')
        eco = str(row.get('ecosystem') or 'unknown')
        strata.setdefault((sig, eco), []).append(dict(row))

    total_pop = len(population_rows)
    target_n = min(int(max_cases), total_pop)
    keys = sorted(strata.keys())
    alloc: dict[tuple[str, str], int] = {k: 1 for k in keys} if len(keys) <= target_n else {}
    remaining = target_n - sum(alloc.values())

    if remaining > 0 and keys:
        quotas = {k: (len(strata[k]) / float(total_pop)) * remaining for k in keys}
        for k in keys:
            cap_extra = len(strata[k]) - alloc.get(k, 0)
            add_k = min(cap_extra, int(math.floor(quotas[k])))
            alloc[k] = alloc.get(k, 0) + add_k
        leftover = target_n - sum(alloc.values())
        rem_order = sorted(keys, key=lambda k: (-(quotas[k] - math.floor(quotas[k])), k))
        for k in rem_order:
            if leftover <= 0:
                break
            if alloc.get(k, 0) < len(strata[k]):
                alloc[k] = alloc.get(k, 0) + 1
                leftover -= 1

    sampled: list[dict[str, Any]] = []
    idx_counter = 1
    for k in keys:
        n_stratum = len(strata[k])
        n_take = min(n_stratum, alloc.get(k, 0))
        if n_take <= 0:
            continue
        sorted_items = sorted(
            strata[k],
            key=lambda r: (
                str(r.get('repository_id') or ''),
                str(r.get('vulnerability_interval_id') or ''),
                str(r.get('anchor_id') or ''),
            ),
        )
        chosen = rng.sample(sorted_items, n_take)
        pi_i = n_take / float(n_stratum)
        ht_w = 1.0 / pi_i
        for item in chosen:
            rec = dict(item)
            rec['case_id'] = str(item.get('case_id') or f'M0CASE-{idx_counter:04d}')
            rec['stratum'] = f'{k[0]}|{k[1]}'
            rec['stratum_population_n'] = n_stratum
            rec['stratum_sample_n'] = n_take
            rec['inclusion_probability'] = round(pi_i, 8)
            rec['ht_weight'] = round(ht_w, 8)
            sampled.append(rec)
            idx_counter += 1
    return sampled


def _verify_verbatim_quotes(
    packets_root: Path,
    case_id: str,
    evidence: list[dict[str, str]],
) -> None:
    if not evidence:
        raise ValueError(f'Missing verbatim evidence quote for {case_id}')
    for item in evidence:
        rel = str(item.get('path') or '').strip()
        quote = str(item.get('quote') or '').strip()
        if not rel or not quote:
            raise ValueError(f'Empty path or quote in evidence for {case_id}')
        rel_clean = rel[len('packets/'):] if rel.startswith('packets/') else rel
        target = (packets_root / rel_clean).resolve()
        if not target.is_relative_to(packets_root.resolve()) or not target.is_file():
            raise ValueError(f'Evidence file does not exist inside packets_root for {case_id}: {rel}')
        content = target.read_text(encoding='utf-8', errors='replace')
        if quote not in content:
            raise ValueError(f'Verbatim quote verification failed for {case_id} in {rel}')


def classify_engine_a_temporal(case_doc: dict[str, Any]) -> tuple[str, str, str]:
    pass
    pr_text = str(case_doc.get('pr_timeline_text') or '').strip()
    git_text = str(case_doc.get('git_commits_text') or '').strip()
    unchanged = case_doc.get('outcome_unchanged')
    has_pr = bool(case_doc.get('has_public_pr_signal'))
    advisory_dt = _parse_dt(case_doc.get('advisory_time_utc'))
    fix_dt = _parse_dt(case_doc.get('fix_commit_time_utc') or case_doc.get('pr_merged_at_utc'))

    if not pr_text and not git_text:
        return 'insufficient_public_trace', 'git_commits.txt', (git_text or 'NO_TRACE_AVAILABLE')[:80]

    if unchanged is True or 'manifest unchanged' in pr_text.lower() or 'lockfile unchanged' in git_text.lower():
        return 'unchanged_with_config', 'pr_timeline.txt', pr_text[:80]

    if advisory_dt is not None and fix_dt is not None and fix_dt < advisory_dt:
        return 'discordant_timing', 'git_commits.txt', git_text[:80]

    if not has_pr or 'NO_PUBLIC_PR' in pr_text:
        return 'manual_maintainer_commit_no_pr', 'git_commits.txt', git_text[:80]

    if 'scheduled' in pr_text.lower() or 'renovation' in pr_text.lower():
        return 'scheduled_bot_maintenance', 'pr_timeline.txt', pr_text[:80]

    return 'advisory_precedes_pr_and_merge', 'pr_timeline.txt', pr_text[:80]


def classify_engine_b_actor_rules(case_doc: dict[str, Any]) -> tuple[str, str, str]:
    pass
    pr_text = str(case_doc.get('pr_timeline_text') or '').strip()
    git_text = str(case_doc.get('git_commits_text') or '').strip()
    combined_low = (pr_text + '\n' + git_text).lower()

    if not git_text or 'insufficient' in combined_low or 'missing_snapshot' in combined_low:
        return 'insufficient_public_trace', 'git_commits.txt', (git_text or pr_text)[:80]

    if 'no_security_pr_found' in combined_low or 'docs update only' in combined_low or 'unchanged' in combined_low:
        return 'unchanged_with_config', 'pr_timeline.txt', pr_text[:80]

    if 'clock_discordance' in combined_low or 'predates' in combined_low or 'discordant' in combined_low:
        return 'discordant_timing', 'git_commits.txt', git_text[:80]

    if 'no_public_pr' in combined_low or 'manual bump' in combined_low or 'manual_hotfix' in combined_low:
        return 'manual_maintainer_commit_no_pr', 'git_commits.txt', git_text[:80]

    if 'scheduled' in combined_low or 'renovate' in combined_low or 'weekly dependency' in combined_low:
        return 'scheduled_bot_maintenance', 'pr_timeline.txt', pr_text[:80]

    if 'pr #' in combined_low or 'pull request' in combined_low or 'dependabot' in combined_low:
        return 'advisory_precedes_pr_and_merge', 'pr_timeline.txt', pr_text[:80]

    return 'insufficient_public_trace', 'git_commits.txt', git_text[:80]


def adjudicate_engines(
    cat_a: str,
    cat_b: str,
    case_doc: dict[str, Any],
) -> tuple[str, str]:
    pass
    if cat_a == cat_b:
        return cat_a, f'Engines agree deterministically on {cat_a}.'
    if 'discordant_timing' in {cat_a, cat_b} and (
        'predates' in str(case_doc.get('git_commits_text') or '').lower()
        or case_doc.get('clock_discordant_flag') is True
    ):
        return 'discordant_timing', f'Deterministic priority rule resolved {cat_a} vs {cat_b} via timestamp check.'
    if cat_a != 'insufficient_public_trace':
        return cat_a, f'Deterministic priority rule selected Engine A ({cat_a}) over Engine B ({cat_b}).'
    return cat_b, f'Deterministic fallback rule selected Engine B ({cat_b}) over Engine A ({cat_a}).'


def run_automated_m0_trace_audit(
    sampled_cases: list[dict[str, Any]],
    packets_root: Path | str,
    out_dir: Path | str,
    *,
    mode: str = 'synthetic',
) -> dict[str, Any]:
    pass
    out_path = Path(out_dir).resolve()
    pkt_root = Path(packets_root).resolve()
    if mode == 'synthetic':
        validation.assert_synthetic_mode_isolated([out_path, pkt_root])
        mode_label = 'SYNTHETIC_M0_AUTOMATED_FIXTURE'
    elif mode == 'official':
        validation.assert_official_empirical_unlocked()
        mode_label = 'OFFICIAL_EMPIRICAL_REV4_AUTOMATED'
    else:
        raise ValueError(f'Unsupported mode {mode!r}')

    if len(sampled_cases) > MAX_M0_SAMPLE_CASES:
        raise ValueError(f'M0 sample exceeds cap {MAX_M0_SAMPLE_CASES}: {len(sampled_cases)}')

    out_path.mkdir(parents=True, exist_ok=True)
    final_records: list[dict[str, Any]] = []
    labels_a: list[str] = []
    labels_b: list[str] = []
    weights_list: list[float] = []

    for row in sampled_cases:
        cid = str(row['case_id'])
        trace_json_path = pkt_root / cid / 'trace_record.json'
        if not trace_json_path.is_file():
            raise ValueError(f'Missing blinded M0 trace packet: {trace_json_path}')
        case_doc = json.loads(trace_json_path.read_text(encoding='utf-8'))

        cat_a, file_a, quote_a = classify_engine_a_temporal(case_doc)
        cat_b, file_b, quote_b = classify_engine_b_actor_rules(case_doc)
        if cat_a not in M0_TRACE_CODEBOOK or cat_b not in M0_TRACE_CODEBOOK:
            raise ValueError(f'Invalid automated M0 trace category for {cid}: {(cat_a, cat_b)}')

        _verify_verbatim_quotes(pkt_root, cid, [{'path': f'packets/{cid}/{file_a}', 'quote': quote_a}])
        _verify_verbatim_quotes(pkt_root, cid, [{'path': f'packets/{cid}/{file_b}', 'quote': quote_b}])

        final_cat, rationale = adjudicate_engines(cat_a, cat_b, case_doc)
        clean_note = ' '.join(rationale.split())[:180]
        if study_contract.check_no_forbidden_dashes(clean_note):
            raise ValueError(f'Forbidden dash in automated M0 rationale for {cid}')

        w = float(row['ht_weight'])
        labels_a.append(cat_a)
        labels_b.append(cat_b)
        weights_list.append(w)
        final_records.append(
            {
                'case_id': cid,
                'repository_id': str(row.get('repository_id') or row.get('family_id') or cid),
                'signal_type': str(row.get('signal_type') or 'unknown'),
                'ecosystem': str(row.get('ecosystem') or 'unknown'),
                'inclusion_probability': float(row['inclusion_probability']),
                'ht_weight': w,
                'r1_label': cat_a,
                'r2_label': cat_b,
                'adjudicated': (cat_a != cat_b),
                'final_trace_category': final_cat,
                'note_summary': clean_note,
            }
        )

    kappa_unweighted = compute_cohen_kappa(labels_a, labels_b)
    kappa_weighted = compute_cohen_kappa(labels_a, labels_b, weights=weights_list)
    budget = {
        'prior_v1_hours': 0.0,
        'm0_hours': 0.0,
        'total_person_hours': 0.0,
        'within_240h_budget': True,
        'within_rev4_zero_human_cap': True,
        'rev4_person_hour_cap': M0_PERSON_HOUR_CAP,
    }

    audit_rows: list[dict[str, Any]] = []
    strata_scopes: list[tuple[str, str, str, list[dict[str, Any]]]] = [
        ('pooled', 'ALL', 'ALL', final_records),
    ]
    for eco in sorted({r['ecosystem'] for r in final_records}):
        strata_scopes.append(('ecosystem', 'ALL', eco, [r for r in final_records if r['ecosystem'] == eco]))
    for sig in sorted({r['signal_type'] for r in final_records}):
        strata_scopes.append(('signal_type', sig, 'ALL', [r for r in final_records if r['signal_type'] == sig]))

    for scope_name, sig_lbl, eco_lbl, subset in strata_scopes:
        ht_w = np.asarray([r['ht_weight'] for r in subset], dtype=float)
        clusters = [r['repository_id'] for r in subset]
        for cat in M0_TRACE_CODEBOOK:
            ind = np.asarray([1.0 if r['final_trace_category'] == cat else 0.0 for r in subset], dtype=float)
            n_cat = int(ind.sum())
            w_cat = float(np.dot(ht_w, ind))
            p_hat, se, lo, hi = _weighted_proportion_ci(ind, ht_w, clusters)
            audit_rows.append(
                {
                    'mode_label': mode_label,
                    'stratum_scope': scope_name,
                    'signal_type': sig_lbl,
                    'ecosystem': eco_lbl,
                    'trace_category': cat,
                    'n_cases': n_cat,
                    'weighted_count': round(w_cat, 6),
                    'weighted_proportion': round(p_hat, 6),
                    'se': round(se, 6),
                    'ci95_low': round(lo, 6),
                    'ci95_high': round(hi, 6),
                    'cohen_kappa_r1_r2': kappa_unweighted['kappa'],
                    'weighted_kappa_r1_r2': kappa_weighted['kappa'],
                    'prior_v1_hours': 0.0,
                    'm0_hours': 0.0,
                    'total_v1_m0_hours': 0.0,
                    'within_240h_budget': True,
                    'within_rev4_zero_human_cap': True,
                    'author_type': study_contract.REV4_AUTOMATED_M0_AUTHOR_TYPE,
                    'human_attestation': False,
                    'human_validated_claims_withdrawn': True,
                    'output_label': 'tab:e2_a2_a3_e3_channels',
                }
            )

    csv_path = out_path / 'workflow_trace_audit.csv'
    study_contract.atomic_write_csv(csv_path, WORKFLOW_TRACE_AUDIT_COLUMNS, audit_rows)

    discordant = [r for r in final_records if r['r1_label'] != r['r2_label'] or r['final_trace_category'] == 'discordant_timing']
    md_lines = [
        f'# M0 Automated Deterministic Trace Audit Report ({mode_label})',
        '',
        f'- Author type: `{study_contract.REV4_AUTOMATED_M0_AUTHOR_TYPE}`',
        '- Human attestation: `false` (Recorded-data zero-human automated governance)',
        f'- Human person-hours: `0.0` (cap `0.0`)',
        f'- Sampled cases: `{len(final_records)}` (cap `{MAX_M0_SAMPLE_CASES}`)',
        f'- Engine A vs Engine B unweighted Cohen kappa: `{kappa_unweighted["kappa"]:.4f}` (observed agreement `{kappa_unweighted["p_observed"]:.4f}`)',
        f'- Engine A vs Engine B weighted Cohen kappa: `{kappa_weighted["kappa"]:.4f}`',
        '',
        '## Discordant or Timing-Sensitive Cases',
        '',
        '| Case ID | Signal Type | Ecosystem | Engine A | Engine B | Final Category | Note |',
        '| - | - | - | - | - | - | - |',
    ]
    for r in discordant[:40]:
        md_lines.append(
            f"| {r['case_id']} | {r['signal_type']} | {r['ecosystem']} | {r['r1_label']} | {r['r2_label']} | {r['final_trace_category']} | {r['note_summary']} |"
        )
    if not discordant:
        md_lines.append('| none | ALL | ALL | none | none | none | Zero discordant cases observed |')
    md_text = '\n'.join(md_lines) + '\n'
    if study_contract.check_no_forbidden_dashes(md_text):
        raise ValueError('Forbidden dash detected in generated discordant_cases.md')
    md_path = out_path / 'discordant_cases.md'
    study_contract.atomic_write_text(md_path, md_text)

    summary_path = out_path / 'm0_trace_summary.json'
    summary_payload = {
        'schema_version': 'm0_automated_trace_summary.v1',
        'construct_version': CONSTRUCT_VERSION_M0,
        'mode_label': mode_label,
        'is_synthetic': (mode == 'synthetic'),
        'author_type': study_contract.REV4_AUTOMATED_M0_AUTHOR_TYPE,
        'human_attestation': False,
        'human_validated_claims_withdrawn': True,
        'recorded_at_utc': study_contract.utc_now_iso(),
        'sample_size': len(final_records),
        'population_weight_total': round(sum(weights_list), 6),
        'cohen_kappa_unweighted': kappa_unweighted['kappa'],
        'cohen_kappa_weighted': kappa_weighted['kappa'],
        'agreement_observed': kappa_unweighted['p_observed'],
        'provenance_verified': True,
        'budget_check': budget,
        'pooled_proportions': {
            row['trace_category']: row['weighted_proportion']
            for row in audit_rows
            if row['stratum_scope'] == 'pooled'
        },
        'artifacts': {
            'workflow_trace_audit_csv': str(csv_path),
            'discordant_cases_md': str(md_path),
            'm0_trace_summary_json': str(summary_path),
        },
    }
    study_contract.atomic_write_json(summary_path, summary_payload)
    return summary_payload


def export_m0_packets(
    sampled_cases: list[dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
    out_dir: Path | str,
    *,
    mode: str = 'synthetic',
) -> Path:
    pass
    out_path = Path(out_dir).resolve()
    if mode == 'synthetic':
        validation.assert_synthetic_mode_isolated([out_path])
    packets_root = out_path / 'packets'
    packets_root.mkdir(parents=True, exist_ok=True)
    for row in sampled_cases:
        cid = str(row['case_id'])
        cdir = packets_root / cid
        cdir.mkdir(parents=True, exist_ok=True)
        snap = snapshots[cid]
        pr_txt = str(snap.get('pr_timeline.txt') or '')
        git_txt = str(snap.get('git_commits.txt') or '')
        study_contract.atomic_write_text(cdir / 'pr_timeline.txt', pr_txt)
        study_contract.atomic_write_text(cdir / 'git_commits.txt', git_txt)
        record = {
            'case_id': cid,
            'ecosystem': row.get('ecosystem', ''),
            'signal_type': row.get('signal_type', ''),
            'advisory_id': row.get('advisory_id', ''),
            'advisory_time_utc': row.get('event_time_lower_utc', ''),
            'fix_commit_time_utc': snap.get('fix_commit_time_utc', ''),
            'pr_merged_at_utc': snap.get('pr_merged_at_utc', ''),
            'outcome_unchanged': bool(row.get('outcome_unchanged', False)),
            'has_public_pr_signal': bool(snap.get('has_public_pr_signal', False)),
            'clock_discordant_flag': bool(snap.get('clock_discordant_flag', False)),
            'pr_timeline_text': pr_txt.strip(),
            'git_commits_text': git_txt.strip(),
        }
        study_contract.atomic_write_json(cdir / 'trace_record.json', record)
    return packets_root


def generate_synthetic_automated_m0_fixture(
    out_dir: Path | str,
    *,
    seed: int | None = None,
    n_population: int = 240,
    sample_n: int = 60,
) -> dict[str, Any]:
    pass
    eff_seed = (
        int(seed)
        if seed is not None
        else study_contract.derive_child_seed(study_contract.STUDY_ID, 'M0', 'synthetic_rev4', 0)
    )
    ecosystems = ('npm', 'PyPI', 'Maven')
    signals = ('EPSS_cutover', 'GHSA_review', 'KEV_listing')

    panel_rows: list[dict[str, Any]] = []
    for idx in range(1, n_population + 1):
        eco = ecosystems[(idx - 1) % len(ecosystems)]
        sig = signals[((idx - 1) // len(ecosystems)) % len(signals)]
        repo_num = ((idx - 1) % 30) + 1
        panel_rows.append(
            {
                'repository_id': f'org{repo_num}/repo{repo_num}',
                'project_path': '.' if eco != 'Maven' else 'core',
                'dependency_coordinate': f'pkg:{eco.lower()}/lib-{idx % 17}',
                'vulnerability_interval_id': f'VINT-{idx:04d}',
                'anchor_id': f'ANCH-{idx:04d}',
                'cve_id': f'CVE-2025-{(1000 + (idx % 45)):04d}',
                'advisory_id': f'GHSA-demo-{idx:04d}',
                'signal_type': sig,
                'ecosystem': eco,
                'event_time_lower_utc': '2025-04-10T12:00:00Z',
                'has_public_pr_truth': (idx % 3) != 0,
                'outcome_unchanged': (idx % 4) == 0,
            }
        )

    sampled = sample_m0_cases(panel_rows, max_cases=sample_n, seed=eff_seed)
    snapshots: dict[str, dict[str, Any]] = {}
    for pos, row in enumerate(sampled):
        cid = row['case_id']
        unchanged = bool(row['outcome_unchanged'])
        has_pr = bool(row['has_public_pr_truth'])
        if unchanged:
            pr_text = 'NO_SECURITY_PR_FOUND: bot_config=.github/dependabot.yml present before anchor; manifest unchanged.'
            git_text = 'commit 1111aaa (2025-04-20T09:00:00Z): docs update only, lockfile unchanged.'
            fix_ts = '2025-04-20T09:00:00Z'
            discordant = False
            pr_sig = False
        elif not has_pr:
            pr_text = 'NO_PUBLIC_PR: maintainer committed direct fix without opening a pull request.'
            git_text = 'commit 2222bbb (2025-04-12T14:30:00Z): manual bump in manifest.'
            fix_ts = '2025-04-12T14:30:00Z'
            discordant = False
            pr_sig = False
        elif pos % 11 == 0:
            pr_text = f'PR #41 referencing {row["advisory_id"]} merged 2025-04-08T08:00:00Z.'
            git_text = 'commit 4444ddd (2025-04-08T08:00:00Z): fix commit predates advisory clock_discordance.'
            fix_ts = '2025-04-08T08:00:00Z'
            discordant = True
            pr_sig = True
        elif pos % 7 == 0:
            pr_text = f'Scheduled Renovate weekly dependency update PR #43 merged 2025-04-12T11:00:00Z for {row["advisory_id"]}.'
            git_text = 'commit 5555eee (2025-04-12T11:00:00Z): scheduled renovate dependency bump.'
            fix_ts = '2025-04-12T11:00:00Z'
            discordant = False
            pr_sig = True
        else:
            pr_text = f'Dependabot PR #42 opened 2025-04-11T10:00:00Z referencing {row["advisory_id"]} merged 2025-04-12T15:00:00Z.'
            git_text = 'commit 3333ccc (2025-04-12T15:00:00Z): lock and manifest updated to patched version via Dependabot PR #42.'
            fix_ts = '2025-04-12T15:00:00Z'
            discordant = False
            pr_sig = True

        snapshots[cid] = {
            'pr_timeline.txt': pr_text + '\n',
            'git_commits.txt': git_text + '\n',
            'fix_commit_time_utc': fix_ts,
            'pr_merged_at_utc': fix_ts,
            'has_public_pr_signal': pr_sig,
            'clock_discordant_flag': discordant,
        }

    out_path = Path(out_dir).resolve()
    packets_root = export_m0_packets(sampled, snapshots, out_path, mode='synthetic')
    summary = run_automated_m0_trace_audit(sampled, packets_root, out_path, mode='synthetic')
    return {
        'sampled_cases': sampled,
        'packets_root': str(packets_root),
        'summary': summary,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Recorded-data deterministic M0 automated workflow trace audit.')
    parser.add_argument('-m', dest='mode', choices=('synthetic',), default='synthetic', help='Execution mode')
    parser.add_argument('-o', dest='out_dir', required=True, help='Output directory')
    parser.add_argument('-s', dest='seed', type=int, default=study_contract.MASTER_SEED, help='Deterministic seed')
    parser.add_argument('-n', dest='sample_n', type=int, default=60, help='Sample size (<= 120)')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    res = generate_synthetic_automated_m0_fixture(args.out_dir, seed=args.seed, sample_n=args.sample_n)
    print(json.dumps(res['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
