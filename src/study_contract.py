"""Schemas, deterministic seeds and validation of observed data."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

STUDY_ID = 'tenure1_paper4_git_repair_v3'
MASTER_SEED = 20261112

REV4_REVISION = 4
REV4_AUTOMATED_V1_AUTHOR_TYPE = 'automated_blinded_multi_agent_audit'
REV4_AUTOMATED_M0_AUTHOR_TYPE = 'deterministic_automated_trace_classifier'
REV4_FC09_WITHDRAWN_STATUS = 'WITHDRAWN_NO_PERSONAL_OR_DEPOSIT_CLAIM_REV4'

OBSERVATION_KEYS = (
    'repository_id',
    'project_path',
    'dependency_coordinate',
    'vulnerability_interval_id',
    'anchor_id',
)

OBSERVATION_COLUMNS = OBSERVATION_KEYS + (
    'dependency_scope',
    'ecosystem',
    'family_id',
    'cve_id',
    'fix_dyad_id',
    'baseline_commit',
    'before_commit',
    'after_commit',
    'file_path',
    'blob_sha256',
    'source_snapshot_sha256',
    'parser_version',
    'event_time_lower_utc',
    'event_time_upper_utc',
    'state',
    'eligibility_only',
    'observation_status',
    'audit_inclusion_probability',
)

STATE_EXTRA_COLUMNS = (
    'observation_week_utc',
    'major_change',
    'out_of_range',
    'successor_coordinate',
    'successor_safe_status',
    'lock_evidence',
    'manifest_evidence',
    'unknown_reason',
)

STATE_COLUMNS = OBSERVATION_COLUMNS + STATE_EXTRA_COLUMNS

FIRST_EVENT_EXTRA_COLUMNS = (
    'first_event_time',
    'first_cause',
    'censor_time',
    'censor_reason',
    'reentry_flag',
)

FIRST_EVENT_COLUMNS = OBSERVATION_COLUMNS + FIRST_EVENT_EXTRA_COLUMNS

RESULT_ROW_COLUMNS = (
    'experiment_id',
    'run_id',
    'config_sha256',
    'code_sha256',
    'input_sha256',
    'estimand',
    'exposure',
    'comparator',
    'subgroup',
    'horizon_days',
    'estimate',
    'lower',
    'upper',
    'scale',
    'uncertainty_method',
    'independent_unit',
    'n_repositories',
    'n_cves',
    'n_fix_dyads',
    'n_calendar_shocks',
    'multiplicity_family',
    'test_status',
    'failure_reason',
)

NONESTIMABLE_STATUSES = {
    'NOT_ESTIMABLE',
    'NOT_IDENTIFIED',
    'INSUFFICIENT_CLUSTERS',
    'SPARSE_ZERO_CELL',
    'BLOCKED',
}

ESTIMABLE_STATUSES = {
    'ESTIMABLE',
    'DIAGNOSTIC_ONLY',
    'LIMITED_INFERENCE_FEW_CLUSTERS',
}

VALID_OBSERVATION_STATUSES = {
    'observed',
    'unknown',
    'censored',
    'eligibility_only',
}

VALID_STATES = {None, 0, 1, 2, 3, 4}
VALID_FIRST_CAUSES = {None, 1, 2, 3, 4}

FORBIDDEN_AI_ACTOR_TOKENS = (
    'llm_surrogate',
    'llm_swarm',
    'surrogate_reviewer',
    'surrogate_adjudicator',
    'surrogate',
    'ai_surrogate',
    'ai_reviewer',
    'ai_generated',
    'synthetic_human',
    'claude',
    'codex',
    'chatgpt',
    'gpt-4',
    'gpt-5',
    'openai',
    'anthropic',
    'gemini',
    'swarm_worker',
    'subagent',
)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


_HEX_64_CHARS = frozenset('0123456789abcdef')


def validate_sha256_hex(value: Any, context: str = 'sha256') -> str:
    pass
    if not isinstance(value, str):
        raise ValueError(f'Missing or invalid 64-hex sha256 in {context}: {value!r}')
    text = value.strip().lower()
    if len(text) != 64 or any(ch not in _HEX_64_CHARS for ch in text):
        raise ValueError(f'Missing or invalid 64-hex sha256 in {context}: {value!r}')
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=str(target.parent),
            prefix=f'.{target.name}.',
            suffix='.tmp',
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n'
    else:
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    atomic_write_text(path, text)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    pass
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=str(target.parent),
            prefix=f'.{target.name}.',
            suffix='.tmp',
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, separators=(',', ':'), ensure_ascii=False, allow_nan=False))
                handle.write('\n')
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        return count
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            newline='',
            dir=str(target.parent),
            prefix=f'.{target.name}.',
            suffix='.tmp',
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                clean = {}
                for key in fieldnames:
                    val = row.get(key)
                    if isinstance(val, (dict, list, tuple)):
                        clean[key] = json.dumps(val, ensure_ascii=False)
                    elif val is None:
                        clean[key] = ''
                    else:
                        clean[key] = val
                writer.writerow(clean)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def check_no_forbidden_dashes(text: str) -> list[tuple[int, str]]:
    pass
    findings: list[tuple[int, str]] = []
    double_hyphen = '-' * 2
    for lineno, line in enumerate(text.splitlines(), start=1):
        if '\u2014' in line or '\u2013' in line or double_hyphen in line:
            findings.append((lineno, line[:120]))
    return findings


def check_no_ai_actor_tokens(text: str, field_name: str) -> None:
    pass
    low = str(text or '').lower()
    for tok in FORBIDDEN_AI_ACTOR_TOKENS:
        if '_' in tok or '-' in tok:
            if tok in low:
                raise ValueError(f'SEOS provenance violation in {field_name}: forbidden actor token {tok!r}')
        else:
            if re.search(r'(?<![a-z0-9])' + re.escape(tok) + r'(?![a-z0-9])', low):
                raise ValueError(f'SEOS provenance violation in {field_name}: forbidden actor token {tok!r}')


def derive_child_seed(
    study_id: str,
    experiment_id: str,
    scenario: str,
    replicate: int = 0,
) -> int:
    pass
    payload = f'{MASTER_SEED}|{study_id}|{experiment_id}|{scenario}|{int(replicate)}'
    digest = hashlib.sha256(payload.encode('utf-8')).digest()
    seed = int.from_bytes(digest[:4], 'big')
    return seed if seed > 0 else 1


def paired_cluster_resampling_indices(n_clusters: int, reps: int, seed: int) -> np.ndarray:
    pass
    if n_clusters < 2:
        raise ValueError('At least two independent clusters required for paired resampling')
    if reps < 1:
        raise ValueError('Bootstrap/resampling replicates must be positive')
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, n_clusters, size=(int(reps), int(n_clusters)))


def _require_columns(row: dict[str, Any], required: tuple[str, ...], label: str, idx: int) -> None:
    missing = [col for col in required if col not in row]
    if missing:
        raise ValueError(f'{label}[{idx}] missing required columns: {missing}')


def _validate_state_and_eligibility(row: dict[str, Any], label: str, idx: int) -> None:
    status = row.get('observation_status')
    if status not in VALID_OBSERVATION_STATUSES:
        raise ValueError(f'{label}[{idx}] invalid observation_status {status!r}')
    state = row.get('state')
    if isinstance(state, bool) or state not in VALID_STATES:
        raise ValueError(f'{label}[{idx}] invalid state {state!r}')
    if status == 'unknown' and state is not None:
        raise ValueError(f'{label}[{idx}] unknown observation_status must keep state=None (never 0)')
    if not isinstance(row.get('eligibility_only'), bool):
        raise ValueError(f'{label}[{idx}] eligibility_only must be bool')
    prob = row.get('audit_inclusion_probability')
    if prob is not None:
        p = float(prob)
        if not math.isfinite(p) or p <= 0.0 or p > 1.0:
            raise ValueError(f'{label}[{idx}] audit_inclusion_probability must be in (0, 1]')


def validate_observation_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass
    seen_keys: set[tuple[Any, ...]] = set()
    for idx, row in enumerate(rows):
        _require_columns(row, OBSERVATION_COLUMNS, 'observation_table', idx)
        _validate_state_and_eligibility(row, 'observation_table', idx)
        key = tuple(row[k] for k in OBSERVATION_KEYS)
        if any(v is None or v == '' for v in key):
            raise ValueError(f'observation_table[{idx}] contains empty key component: {key!r}')
        if key in seen_keys:
            raise ValueError(f'observation_table duplicate natural key (join multiplicity bug): {key!r}')
        seen_keys.add(key)
    return {'row_count': len(rows), 'unique_keys': len(seen_keys), 'valid': True}


def validate_state_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass
    seen_keys: set[tuple[Any, ...]] = set()
    for idx, row in enumerate(rows):
        _require_columns(row, STATE_COLUMNS, 'state_table', idx)
        _validate_state_and_eligibility(row, 'state_table', idx)
        if row['state'] is None and not str(row.get('unknown_reason') or '').strip():
            raise ValueError(f'state_table[{idx}] state=None requires non-empty unknown_reason')
        key = tuple(row[k] for k in OBSERVATION_KEYS) + (row['observation_week_utc'],)
        if key in seen_keys:
            raise ValueError(f'state_table duplicate weekly observation key: {key!r}')
        seen_keys.add(key)
    return {'row_count': len(rows), 'unique_week_keys': len(seen_keys), 'valid': True}


def validate_first_event_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pass
    seen_keys: set[tuple[Any, ...]] = set()
    for idx, row in enumerate(rows):
        _require_columns(row, FIRST_EVENT_COLUMNS, 'first_event_table', idx)
        _validate_state_and_eligibility(row, 'first_event_table', idx)
        key = tuple(row[k] for k in OBSERVATION_KEYS)
        if key in seen_keys:
            raise ValueError(f'first_event_table duplicate natural key: {key!r}')
        seen_keys.add(key)
        cause = row.get('first_cause')
        if isinstance(cause, bool) or cause not in VALID_FIRST_CAUSES:
            raise ValueError(f'first_event_table[{idx}] invalid first_cause {cause!r}')
        if cause is None and not str(row.get('censor_reason') or '').strip():
            raise ValueError(f'first_event_table[{idx}] censored/unknown row requires censor_reason')
        if not isinstance(row.get('reentry_flag'), bool):
            raise ValueError(f'first_event_table[{idx}] reentry_flag must be bool')
    return {'row_count': len(rows), 'unique_keys': len(seen_keys), 'valid': True}


def validate_result_row(row: dict[str, Any]) -> dict[str, Any]:
    pass
    missing = [col for col in RESULT_ROW_COLUMNS if col not in row]
    if missing:
        raise ValueError(f'result_row missing required columns: {missing}')
    status = str(row.get('test_status') or '')
    est = row.get('estimate')
    lo = row.get('lower')
    hi = row.get('upper')
    if status in NONESTIMABLE_STATUSES:
        if est is not None or lo is not None or hi is not None:
            raise ValueError(f'Nonestimable result row {row.get("experiment_id")} ({status}) must keep estimate/lower/upper as None')
        if not str(row.get('failure_reason') or '').strip():
            raise ValueError(f'Nonestimable result row {row.get("experiment_id")} ({status}) requires failure_reason')
    elif status in ESTIMABLE_STATUSES:
        for name, val in (('estimate', est), ('lower', lo), ('upper', hi)):
            if val is None or isinstance(val, bool) or not math.isfinite(float(val)):
                raise ValueError(f'Estimable result row {row.get("experiment_id")} invalid {name}: {val!r}')
        if float(lo) > float(est) + 1e-12 or float(est) > float(hi) + 1e-12:
            raise ValueError(f'Estimable result row {row.get("experiment_id")} violated lower <= estimate <= upper: {(lo, est, hi)}')
    else:
        raise ValueError(f'Unknown test_status {status!r}')
    return row
