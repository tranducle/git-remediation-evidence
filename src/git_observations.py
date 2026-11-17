"""Construct Git observations and first-transition outcomes from immutable snapshots."""
from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import threading
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
EXECUTION_ROOT = Path(__file__).resolve().parent.parent
STUDY_ROOT = Path(__file__).resolve().parent.parent
IMPL_V4_DIR = Path(__file__).resolve().parent
I0_DIR = Path(__file__).resolve().parent

for _path in (str(HERE), str(IMPL_V4_DIR), str(I0_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import construct_rules              
import study_contract              
import estimators              
import validation              
import history_measurement as hm              
import version_rules              

STUDY_START_UTC = dt.datetime(2021, 5, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
STUDY_END_UTC = dt.datetime(2026, 10, 31, 23, 59, 59, tzinfo=dt.timezone.utc)
HORIZON_DAYS_MAP = {
    '91d': 91,
    '90d': 90,
    '180d': 180,
    '182d': 182,
    '364d': 364,
}
DEFAULT_DURABLE_WORKERS = 2
MAX_DURABLE_WORKERS = 8
MAX_NEW_DISK_GIB = 40.0
MIN_FREE_DISK_GIB = 20.0
MAX_COLLECTION_WALL_SECONDS = 72 * 3600
MAX_AGGREGATE_RSS_GIB = 10.0
MAX_WORKER_RSS_GIB = 4.0

_BLOB_CACHE_LOCK = threading.Lock()
_BLOB_CACHE: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_BLOB_CACHE_BYTES = 0
_BLOB_CACHE_MAX_BYTES = 256 * 1024 * 1024

_PARSED_SNAPSHOT_LOCK = threading.Lock()
_PARSED_SNAPSHOT_CACHE: OrderedDict[
    tuple[str, tuple[tuple[str, str], ...]],
    tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]],
] = OrderedDict()
_PARSED_SNAPSHOT_CACHE_MAX = 4096

LINEAGE_COLUMNS = (
    'repository_id',
    'project_path',
    'dependency_coordinate',
    'vulnerability_interval_id',
    'anchor_id',
    'ecosystem',
    'family_id',
    'baseline_commit',
    'endpoint_commit',
    'file_path',
    'blob_sha256',
    'source_snapshot_sha256',
    'parser_version',
)

FAILURE_COLUMNS = (
    'repository_id',
    'project_path',
    'dependency_coordinate',
    'vulnerability_interval_id',
    'anchor_id',
    'ecosystem',
    'stage',
    'failure_reason',
    'observation_status',
)


def parse_utc(value: Any) -> dt.datetime | None:
    pass
    if value is None or value == '':
        return None
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.timezone.utc) if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    parsed = dt.datetime.fromisoformat(text.replace('Z', '+00:00'))
    return parsed.astimezone(dt.timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _read_verified_blob(blob_path: str | Path, expected_sha256: str = '') -> bytes:
    pass
    global _BLOB_CACHE_BYTES
    resolved = Path(blob_path).resolve()
    exp_sha = study_contract.validate_sha256_hex(expected_sha256, f'blob {resolved}')
    cache_key = (str(resolved), exp_sha)
    with _BLOB_CACHE_LOCK:
        cached = _BLOB_CACHE.get(cache_key)
        if cached is not None:
            _BLOB_CACHE.move_to_end(cache_key)
            return cached

    raw = resolved.read_bytes()
    actual_sha = study_contract.sha256_bytes(raw)
    if actual_sha != exp_sha:
        raise ValueError(f'Blob SHA-256 mismatch for {resolved}: expected {exp_sha}, got {actual_sha}')

    blob_size = len(raw)
    if blob_size <= _BLOB_CACHE_MAX_BYTES // 4:
        with _BLOB_CACHE_LOCK:
            existing = _BLOB_CACHE.get(cache_key)
            if existing is not None:
                _BLOB_CACHE.move_to_end(cache_key)
                return existing
            _BLOB_CACHE[cache_key] = raw
            _BLOB_CACHE_BYTES += blob_size
            while _BLOB_CACHE_BYTES > _BLOB_CACHE_MAX_BYTES and _BLOB_CACHE:
                _, evicted = _BLOB_CACHE.popitem(last=False)
                _BLOB_CACHE_BYTES -= len(evicted)
    return raw


def _normalize_snapshot(snap: dict[str, Any] | None) -> dict[str, Any] | None:
    pass
    if snap is None:
        return None
    if snap.get('missing') or snap.get('deleted'):
        return None
    raw_files_in = snap.get('raw_files') if 'raw_files' in snap else snap.get('files', {})
    if not isinstance(raw_files_in, dict):
        return None
    files_bytes: dict[str, bytes] = {}
    provenance_in = dict(snap.get('provenance') or {})
    provenance_out: dict[str, dict[str, Any]] = {}
    for rel_path, content in raw_files_in.items():
        if isinstance(content, dict):
            if '__bytes_b64__' in content:
                raw = base64.b64decode(str(content['__bytes_b64__']).encode('ascii'))
            elif '__blob_path__' in content:
                raw = _read_verified_blob(content['__blob_path__'], str(content.get('sha256') or ''))
            else:
                raise TypeError(f'Unsupported snapshot file marker keys for {rel_path}: {sorted(content.keys())}')
        elif isinstance(content, str):
            raw = content.encode('utf-8')
        else:
            raw = bytes(content)
        files_bytes[str(rel_path)] = raw
        entry = dict(provenance_in.get(rel_path) or {})
        if isinstance(content, dict) and content.get('sha256'):
            entry.setdefault('content_sha256', str(content['sha256']))
        else:
            entry.setdefault('content_sha256', study_contract.sha256_bytes(raw))
        provenance_out[str(rel_path)] = entry
    epoch = snap.get('commit_epoch')
    if epoch is None and snap.get('commit_utc'):
        parsed_dt = parse_utc(snap['commit_utc'])
        epoch = parsed_dt.timestamp() if parsed_dt is not None else None
    return {
        'commit_oid': str(snap['commit_oid']),
        'commit_epoch': epoch,
        'files': files_bytes,
        'provenance': provenance_out,
        'bot_config_paths': list(snap.get('bot_config_paths', [])),
    }


def _snapshot_manifest_sha256(snapshot: dict[str, Any] | None) -> str:
    pass
    if not snapshot:
        return ''
    items = [
        {
            'path': path,
            'sha256': snapshot['provenance'][path]['content_sha256'],
        }
        for path in sorted(snapshot['files'])
    ]
    payload = json.dumps({'commit_oid': snapshot['commit_oid'], 'files': items}, sort_keys=True).encode('utf-8')
    return study_contract.sha256_bytes(payload)


def _primary_source_metadata(proj: dict[str, Any] | None, snapshot: dict[str, Any] | None) -> tuple[str, str]:
    pass
    if proj and proj.get('source_objects'):
        sources = proj['source_objects']
        chosen_path = proj['project_path'] if proj['project_path'] in sources else sorted(sources)[0]
        meta = sources.get(chosen_path) or {}
        digest = str(meta.get('content_sha256') or meta.get('sha256') or '')
        if not digest and snapshot and chosen_path in snapshot['files']:
            digest = study_contract.sha256_bytes(snapshot['files'][chosen_path])
        return str(chosen_path), digest
    if snapshot and snapshot['files']:
        chosen_path = sorted(snapshot['files'])[0]
        return chosen_path, snapshot['provenance'][chosen_path]['content_sha256']
    return '', ''


def _select_project(
    snapshot: dict[str, Any] | None,
    ecosystem: str,
    expected_project: str | None,
    project_path_hint: str | None = None,
    pom_store: Any = None,
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    pass
    if snapshot is None:
        return None
    files_map = snapshot.get('files')
    prov_map = snapshot.get('provenance')
    can_memoize = (
        pom_store is None
        and isinstance(files_map, dict)
        and isinstance(prov_map, dict)
        and all(
            isinstance(prov_map.get(path), dict) and bool(prov_map[path].get('content_sha256'))
            for path in files_map
        )
    )
    if can_memoize:
        files_sig = tuple((str(path), str(prov_map[path]['content_sha256'])) for path in sorted(files_map))
        cache_key = (str(ecosystem), files_sig)
        with _PARSED_SNAPSHOT_LOCK:
            cached = _PARSED_SNAPSHOT_CACHE.get(cache_key)
            if cached is not None:
                _PARSED_SNAPSHOT_CACHE.move_to_end(cache_key)
        if cached is None:
            raw_diags: list[dict[str, Any]] = []
            raw_projs = hm.parse_projects(
                snapshot,
                ecosystem,
                expected_name=None,
                pom_store=None,
                diagnostics=raw_diags,
            )
            with _PARSED_SNAPSHOT_LOCK:
                existing = _PARSED_SNAPSHOT_CACHE.get(cache_key)
                if existing is not None:
                    _PARSED_SNAPSHOT_CACHE.move_to_end(cache_key)
                    cached = existing
                else:
                    cached = (tuple(raw_projs), tuple(raw_diags))
                    _PARSED_SNAPSHOT_CACHE[cache_key] = cached
                    while len(_PARSED_SNAPSHOT_CACHE) > _PARSED_SNAPSHOT_CACHE_MAX:
                        _PARSED_SNAPSHOT_CACHE.popitem(last=False)

        cached_projs, cached_diags = cached
        if diagnostics is not None and cached_diags:
            diagnostics.extend(dict(diag) for diag in cached_diags)

        parsed: list[dict[str, Any]] = []
        norm_exp: str | None = None
        bot_paths_now = list(snapshot.get('bot_config_paths', []))
        commit_oid_now = snapshot['commit_oid']
        commit_epoch_now = snapshot.get('commit_epoch')
        for tmpl in cached_projs:
            if expected_project is not None:
                pname = tmpl.get('project_name')
                if not pname:
                    continue
                if norm_exp is None:
                    norm_exp = hm.normalize(expected_project, ecosystem)
                if hm.normalize(pname, ecosystem) != norm_exp:
                    continue
            p_copy = dict(tmpl)
            p_copy['commit_oid'] = commit_oid_now
            p_copy['commit_epoch'] = commit_epoch_now
            p_copy['bot_config_paths'] = list(bot_paths_now)
            p_copy['source_objects'] = {
                src_p: prov_map.get(src_p, meta)
                for src_p, meta in tmpl['source_objects'].items()
            }
            parsed.append(p_copy)
    else:
        parsed = hm.parse_projects(
            snapshot,
            ecosystem,
            expected_name=expected_project,
            pom_store=pom_store,
            diagnostics=diagnostics,
        )
    if not parsed:
        return None
    if len(parsed) == 1:
        return parsed[0]
    if project_path_hint:
        for cand in parsed:
            if cand.get('project_path') == project_path_hint:
                return cand
    return None


def _compute_reentry_flag(states: list[int | None]) -> tuple[bool, dict[str, int | None]]:
    pass
    summary = estimators.reversible_and_first(states)
    reentered = False
    for idx in range(1, len(states) + 1):
        prefix = estimators.reversible_and_first(states[:idx])
        if prefix['first_response_state'] is not None and prefix['current_state'] == 0:
            reentered = True
            break
    return reentered, summary


def _compute_out_of_range(
    baseline_proj: dict[str, Any] | None,
    dependency_coordinate: str,
    fixed_version: str,
    ecosystem: str,
) -> bool | None:
    pass
    if not baseline_proj:
        return None
    norm_pkg = hm.normalize(dependency_coordinate, ecosystem)
    spec = baseline_proj.get('dependencies', {}).get(norm_pkg)
    if not spec:
        return None
    permits = version_rules.allows(spec, fixed_version, ecosystem)
    if permits is None:
        return None
    return not bool(permits)


def _compute_major_change(
    baseline_obs: dict[str, Any] | None,
    after_obs: dict[str, Any] | None,
    ecosystem: str,
    tr_row: dict[str, Any] | None = None,
) -> bool | None:
    pass
    if tr_row and tr_row.get('major_change') is not None:
        return bool(tr_row['major_change'])
    if not baseline_obs or not after_obs:
        return None
    old_ver = baseline_obs.get('version')
    new_ver = after_obs.get('version')
    if old_ver and new_ver:
        res = version_rules.major_change(old_ver, new_ver, ecosystem)
        if res is not None:
            return bool(res)
    old_c = baseline_obs.get('constraint')
    new_c = after_obs.get('constraint')
    if old_c and new_c:
        old_m = construct_rules.constraint_major(old_c, ecosystem)
        new_m = construct_rules.constraint_major(new_c, ecosystem)
        if old_m is not None and new_m is not None:
            return bool(old_m != new_m)
    return None


def _is_eligibility_only(
    state: int | None,
    proj: dict[str, Any] | None,
    dependency_coordinate: str,
    fixed_version: str,
    ecosystem: str,
    fix_released_dt: dt.datetime | None,
    snap_dt: dt.datetime | None,
    legacy_release_eligible: bool = False,
) -> bool:
    pass
    if state != 0:
        return False
    upstream_released = (
        fix_released_dt is not None
        and snap_dt is not None
        and snap_dt >= fix_released_dt
    )
    if not upstream_released and not legacy_release_eligible:
        return False
    if legacy_release_eligible:
        return True
    if not proj:
        return False
    norm_pkg = hm.normalize(dependency_coordinate, ecosystem)
    spec = proj.get('dependencies', {}).get(norm_pkg)
    if not spec:
        return False
    return version_rules.allows(spec, fixed_version, ecosystem) is True


def build_durable_collection_job_config(
    job_id: str,
    job_key: str,
    run_dir: Path | str,
    out_dir: Path | str,
    cwd: Path | str | None = None,
    max_workers: int = DEFAULT_DURABLE_WORKERS,
    max_new_disk_gib: float = MAX_NEW_DISK_GIB,
    min_free_disk_gib: float = MIN_FREE_DISK_GIB,
    max_wall_seconds: int = MAX_COLLECTION_WALL_SECONDS,
    aggregate_rss_gib: float = MAX_AGGREGATE_RSS_GIB,
    worker_rss_gib: float = MAX_WORKER_RSS_GIB,
    mode: str = 'official',
) -> dict[str, Any]:
    pass
    if not str(job_id or '').strip() or not str(job_key or '').strip():
        raise ValueError('job_id and job_key must be non-empty strings')
    if int(max_workers) < 1 or int(max_workers) > MAX_DURABLE_WORKERS:
        raise ValueError(f'FC-10 violation: max_workers={max_workers} exceeds cap {MAX_DURABLE_WORKERS}')
    if float(max_new_disk_gib) <= 0.0 or float(max_new_disk_gib) > MAX_NEW_DISK_GIB:
        raise ValueError(f'FC-10 violation: max_new_disk_gib={max_new_disk_gib} exceeds ceiling {MAX_NEW_DISK_GIB}')
    if float(min_free_disk_gib) < MIN_FREE_DISK_GIB:
        raise ValueError(f'FC-10 violation: min_free_disk_gib={min_free_disk_gib} below floor {MIN_FREE_DISK_GIB}')
    if int(max_wall_seconds) <= 0 or int(max_wall_seconds) > MAX_COLLECTION_WALL_SECONDS:
        raise ValueError(f'FC-10 violation: max_wall_seconds={max_wall_seconds} exceeds 72h cap')
    if float(aggregate_rss_gib) <= 0.0 or float(aggregate_rss_gib) > MAX_AGGREGATE_RSS_GIB:
        raise ValueError(f'FC-10 violation: aggregate_rss_gib={aggregate_rss_gib} exceeds {MAX_AGGREGATE_RSS_GIB}')
    if float(worker_rss_gib) <= 0.0 or float(worker_rss_gib) > MAX_WORKER_RSS_GIB:
        raise ValueError(f'FC-10 violation: worker_rss_gib={worker_rss_gib} exceeds {MAX_WORKER_RSS_GIB}')
    if mode not in {'official', 'synthetic'}:
        raise ValueError(f'Invalid mode {mode!r}')

    work_dir = str(Path(cwd or STUDY_ROOT).resolve())
    run_root = str(Path(run_dir).resolve())
    dest_dir = str(Path(out_dir).resolve())
    command = [
        sys.executable,
        str((HERE / 'git_observations.py').resolve()),
        '-m',
        mode,
        '-o',
        dest_dir,
        '-w',
        str(int(max_workers)),
    ]
    return {
        'job_id': str(job_id),
        'job_key': str(job_key),
        'run_dir': run_root,
        'cwd': work_dir,
        'command': command,
        'max_wall_seconds': int(max_wall_seconds),
        'parameters': {
            'mode': mode,
            'out_dir': dest_dir,
            'max_workers': int(max_workers),
            'max_new_disk_gib': float(max_new_disk_gib),
            'min_free_disk_gib': float(min_free_disk_gib),
            'aggregate_rss_gib': float(aggregate_rss_gib),
            'worker_rss_gib': float(worker_rss_gib),
            'parser_version': hm.PARSER_VERSION,
        },
    }


def _evaluate_snapshot_step(
    baseline_proj: dict[str, Any] | None,
    prev_valid_proj: dict[str, Any] | None,
    prev_obs: dict[str, Any] | None,
    prev_state: int | None,
    after_proj: dict[str, Any] | None,
    dependency: str,
    introduced: str,
    fixed: str,
    verified_aliases: list[dict[str, Any]],
) -> tuple[int | None, dict[str, Any] | None, dict[str, Any], str]:
    pass
    if baseline_proj is None or after_proj is None:
        return None, None, {}, 'missing_parsed_project_snapshot'
    new_obs = hm.observation(after_proj, dependency, introduced, fixed)
    base_tr = hm.transition(
        baseline_proj,
        after_proj,
        dependency,
        introduced,
        fixed,
        verified_aliases=verified_aliases,
    )
    step_tr = (
        hm.transition(
            prev_valid_proj,
            after_proj,
            dependency,
            introduced,
            fixed,
            verified_aliases=verified_aliases,
        )
        if prev_valid_proj is not None
        else base_tr
    )
    if prev_obs is not None and prev_obs.get('affected') is False and new_obs.get('affected') is True:
        return 0, new_obs, step_tr, str(step_tr.get('reason') or 'recorded reversion restores current exposure')
    if step_tr.get('state') in {1, 2, 3, 4}:
        return int(step_tr['state']), new_obs, step_tr, str(step_tr.get('reason') or '')
    if prev_valid_proj is not None and after_proj['commit_oid'] == prev_valid_proj['commit_oid']:
        if prev_state in {0, 1, 2, 3, 4}:
            return prev_state, new_obs, step_tr, 'unchanged immutable commit retains verified state'
    if prev_state == 4 and new_obs.get('present') is False:
        added_coords = set(after_proj.get('dependencies', {})) - set((prev_valid_proj or {}).get('dependencies', {}))
        if not added_coords:
            return 4, new_obs, step_tr, 'verified removal persisted with no new coordinates'
    if base_tr.get('state') is not None:
        return int(base_tr['state']), new_obs, base_tr, str(base_tr.get('reason') or '')
    reason = str(base_tr.get('reason') or step_tr.get('reason') or 'unclassified_unknown')
    return None, new_obs, base_tr, reason


def _format_csv_row(fieldnames: tuple[str, ...], row: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key in fieldnames:
        val = row.get(key)
        if isinstance(val, (dict, list, tuple)):
            clean[key] = json.dumps(val, ensure_ascii=False)
        elif val is None:
            clean[key] = ''
        else:
            clean[key] = val
    return clean


def build_panel_tables(
    records: Iterable[dict[str, Any]],
    study_end_utc: str | dt.datetime = STUDY_END_UTC,
    pom_store: Any = None,
    stream_state_csv_path: Path | None = None,
) -> dict[str, Any]:
    pass
    state_target = Path(stream_state_csv_path) if stream_state_csv_path is not None else None
    state_tmp_path: Path | None = None
    state_handle: Any = None
    state_writer: csv.DictWriter[str] | None = None

    try:
        if state_target is not None:
            state_target.parent.mkdir(parents=True, exist_ok=True)
            state_handle = tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                newline='',
                dir=str(state_target.parent),
                prefix=f'.{state_target.name}.',
                suffix='.tmp',
                delete=False,
            )
            state_tmp_path = Path(state_handle.name)
            state_writer = csv.DictWriter(state_handle, fieldnames=list(study_contract.STATE_COLUMNS), extrasaction='ignore')
            state_writer.writeheader()

        res = _build_panel_tables_core(
            records,
            study_end_utc=study_end_utc,
            pom_store=pom_store,
            state_writer=state_writer,
        )
        if state_handle is not None and state_tmp_path is not None and state_target is not None:
            state_handle.flush()
            os.fsync(state_handle.fileno())
            state_handle.close()
            state_handle = None
            os.replace(state_tmp_path, state_target)
            state_tmp_path = None
        return res
    except Exception:
        if state_handle is not None:
            try:
                state_handle.close()
            except Exception:
                pass
        if state_tmp_path is not None:
            state_tmp_path.unlink(missing_ok=True)
        raise


def _build_panel_tables_core(
    records: Iterable[dict[str, Any]],
    *,
    study_end_utc: str | dt.datetime = STUDY_END_UTC,
    pom_store: Any = None,
    state_writer: csv.DictWriter[str] | None = None,
) -> dict[str, Any]:
    pass
    study_end_dt = parse_utc(study_end_utc) or STUDY_END_UTC
    obs_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    first_event_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    week_stats_by_key: dict[tuple[Any, ...], dict[str, int]] = {}
    streamed_state_count = 0

    fix_signal_cross: dict[str, int] = {}
    pre_2021_fix_count = 0
    in_window_fix_count = 0
    baseline_valid_by_anchor: dict[str, bool] = {}
    day_90_state_by_anchor: dict[str, int | None] = {}
    horizon_counts: dict[str, dict[str, int]] = {
        h: {'total_units': 0, 'eligible_complete_horizon': 0, 'administratively_censored': 0}
        for h in HORIZON_DAYS_MAP
    }

    for rec in records:
        eco = str(rec['ecosystem'])
        dep = str(rec['dependency_coordinate'])
        expected_proj = rec.get('expected_project')
        introduced = str(rec.get('introduced_version', '0'))
        fixed = str(rec['fixed_version'])
        anchor_dt = parse_utc(rec.get('signal_time_utc') or rec.get('anchor_time_utc'))
        fix_dt = parse_utc(rec.get('fix_released_at') or rec.get('fix_time_utc'))
        if anchor_dt is None:
            raise ValueError(f"Record {rec.get('anchor_id')} missing signal_time_utc")

        fix_year = str(fix_dt.year) if fix_dt is not None else 'unknown'
        sig_year = str(anchor_dt.year)
        cross_key = f'fix_{fix_year}|signal_{sig_year}'
        fix_signal_cross[cross_key] = fix_signal_cross.get(cross_key, 0) + 1
        if fix_dt is not None and fix_dt < STUDY_START_UTC:
            pre_2021_fix_count += 1
        elif fix_dt is not None:
            in_window_fix_count += 1

        for h_label, h_days in HORIZON_DAYS_MAP.items():
            horizon_counts[h_label]['total_units'] += 1
            if anchor_dt + dt.timedelta(days=h_days) <= study_end_dt:
                horizon_counts[h_label]['eligible_complete_horizon'] += 1
            else:
                horizon_counts[h_label]['administratively_censored'] += 1

        base_snap = _normalize_snapshot(rec.get('baseline_snapshot'))
        diagnostics: list[dict[str, Any]] = []
        base_proj = _select_project(
            base_snap,
            eco,
            expected_proj,
            project_path_hint=rec.get('project_path'),
            pom_store=rec.get('pom_store', pom_store),
            diagnostics=diagnostics,
        )
        base_obs = hm.observation(base_proj, dep, introduced, fixed) if base_proj is not None else None
        raw_proj_path = (base_proj or {}).get('project_path')
        if raw_proj_path is None or str(raw_proj_path).strip() in {'', 'None'}:
            raw_proj_path = rec.get('project_path')
        if raw_proj_path is None or str(raw_proj_path).strip() in {'', 'None'}:
            resolved_path = '.'
        else:
            resolved_path = str(raw_proj_path).strip()
        dep_scope = str((base_obs or {}).get('dependency_scope') or rec.get('dependency_scope') or 'unresolved')
        if dep_scope not in construct_rules.SCOPES:
            dep_scope = 'unresolved'
        out_of_range = _compute_out_of_range(base_proj, dep, fixed, eco)
        verified_aliases = list(rec.get('verified_aliases') or [])
        prob = rec.get('audit_inclusion_probability', 1.0)
        legacy_elig = bool(rec.get('legacy_release_eligible', False))

        base_file_path, base_blob_sha = _primary_source_metadata(base_proj, base_snap)
        base_snap_sha = _snapshot_manifest_sha256(base_snap)
        base_commit = base_snap['commit_oid'] if base_snap else str(rec.get('baseline_commit') or 'MISSING_BASELINE')

        baseline_valid = bool(base_obs is not None and base_obs.get('affected') is True)
        baseline_fail_reason = ''
        if not baseline_valid:
            if base_snap is None:
                baseline_fail_reason = str(rec.get('baseline_missing_reason') or 'missing_baseline_git_snapshot')
            elif base_proj is None:
                baseline_fail_reason = 'baseline_project_unparsed_or_ambiguous'
            elif base_obs and base_obs.get('affected') is False:
                baseline_fail_reason = 'baseline_not_affected'
            else:
                baseline_fail_reason = str((base_obs or {}).get('basis') or 'baseline_affected_status_unknown')

        weekly_raw = dict(rec.get('weekly_snapshots') or {})
        extended_raw = dict(rec.get('extended_snapshots') or {})

        prev_valid_proj = base_proj if baseline_valid else None
        prev_obs = base_obs if baseline_valid else None
        prev_state: int | None = 0 if baseline_valid else None
        prev_commit_oid = base_commit

        weekly_states: list[int | None] = []
        weekly_rows_for_unit: list[dict[str, Any]] = []
        w12_ctx: tuple[Any, Any, int | None] | None = None
        first_transition_time_utc: dt.datetime | None = None
        first_transition_lower_utc: dt.datetime | None = None
        first_transition_before_commit = base_commit
        first_transition_after_commit = base_commit

        for week_idx in range(1, 14):
            week_dt = anchor_dt + dt.timedelta(weeks=week_idx)
            week_iso = week_dt.isoformat()
            raw_snap = weekly_raw.get(week_idx)
            if raw_snap is None:
                raw_snap = weekly_raw.get(str(week_idx))

            after_snap = _normalize_snapshot(raw_snap)
            after_proj = None
            state_val: int | None = None
            obs_status = 'unknown'
            unknown_reason = ''
            major_chg: bool | None = None
            elig_only = False
            succ_coord: str | None = None
            succ_safe: bool | None = None
            lock_ev: dict[str, Any] | None = None
            manifest_ev: dict[str, Any] | None = None
            before_commit = prev_commit_oid
            after_commit = after_snap['commit_oid'] if after_snap else ''

            if week_dt > study_end_dt:
                obs_status = 'censored'
                unknown_reason = 'post_study_endpoint_administrative_censoring'
                prev_valid_proj = None
                prev_obs = None
                prev_state = None
            elif not baseline_valid:
                obs_status = 'unknown'
                unknown_reason = baseline_fail_reason
            elif after_snap is None:
                obs_status = 'unknown'
                missing_msg = ''
                if isinstance(raw_snap, dict):
                    missing_msg = str(raw_snap.get('missing_reason') or '')
                unknown_reason = missing_msg or 'missing_git_snapshot_no_forward_fill'
                prev_valid_proj = None
                prev_obs = None
                prev_state = None
            else:
                after_proj = _select_project(
                    after_snap,
                    eco,
                    expected_proj,
                    project_path_hint=resolved_path,
                    pom_store=rec.get('pom_store', pom_store),
                    diagnostics=diagnostics,
                )
                state_val, new_obs, tr_row, reason = _evaluate_snapshot_step(
                    base_proj,
                    prev_valid_proj,
                    prev_obs,
                    prev_state,
                    after_proj,
                    dep,
                    introduced,
                    fixed,
                    verified_aliases,
                )
                if state_val is None:
                    obs_status = 'unknown'
                    unknown_reason = reason or 'unknown_transition_state'
                    prev_valid_proj = None
                    prev_obs = None
                    prev_state = None
                else:
                    snap_dt = parse_utc(after_snap.get('commit_epoch')) or week_dt
                    elig_only = _is_eligibility_only(
                        state_val,
                        after_proj,
                        dep,
                        fixed,
                        eco,
                        fix_dt,
                        snap_dt,
                        legacy_release_eligible=legacy_elig,
                    )
                    obs_status = 'eligibility_only' if elig_only else 'observed'
                    major_chg = _compute_major_change(base_obs, new_obs, eco, tr_row=tr_row)
                    if state_val in {1, 2} and after_proj is not None:
                        norm_dep = hm.normalize(dep, eco)
                        if norm_dep not in after_proj.get('dependencies', {}):
                            for al in verified_aliases:
                                if al.get('old') == dep and al.get('new') in after_proj.get('dependencies', {}):
                                    succ_coord = str(al.get('new'))
                                    succ_safe = True
                                    break
                    if after_proj is not None:
                        norm_dep = hm.normalize(dep, eco)
                        lock_ver = after_proj.get('locked_versions', {}).get(norm_dep)
                        lock_ev = {'locked_version': lock_ver} if lock_ver else None
                        man_decl = after_proj.get('dependencies', {}).get(norm_dep)
                        manifest_ev = {'constraint': man_decl} if man_decl is not None else None
                    if state_val in {1, 2, 3, 4} and first_transition_time_utc is None:
                        first_transition_time_utc = snap_dt
                        first_transition_lower_utc = parse_utc((prev_valid_proj or {}).get('commit_epoch')) or anchor_dt
                        first_transition_before_commit = before_commit
                        first_transition_after_commit = after_commit
                    prev_valid_proj = after_proj
                    prev_obs = new_obs
                    prev_state = state_val
                    prev_commit_oid = after_commit

            snap_file_path, snap_blob_sha = _primary_source_metadata(after_proj or base_proj, after_snap or base_snap)
            snap_sha = _snapshot_manifest_sha256(after_snap) or base_snap_sha
            lower_utc = anchor_dt.isoformat()
            upper_utc = min(week_dt, study_end_dt).isoformat()

            week_row = {
                'repository_id': str(rec['repository_id']),
                'project_path': resolved_path,
                'dependency_coordinate': dep,
                'vulnerability_interval_id': str(rec['vulnerability_interval_id']),
                'anchor_id': str(rec['anchor_id']),
                'dependency_scope': dep_scope,
                'ecosystem': eco,
                'family_id': str(rec['family_id']),
                'cve_id': str(rec['cve_id']),
                'fix_dyad_id': str(rec['fix_dyad_id']),
                'baseline_commit': base_commit,
                'before_commit': before_commit,
                'after_commit': after_commit,
                'file_path': snap_file_path or base_file_path,
                'blob_sha256': snap_blob_sha or base_blob_sha,
                'source_snapshot_sha256': snap_sha,
                'parser_version': hm.PARSER_VERSION,
                'event_time_lower_utc': lower_utc,
                'event_time_upper_utc': upper_utc,
                'state': state_val,
                'eligibility_only': bool(elig_only),
                'observation_status': obs_status,
                'audit_inclusion_probability': prob,
                'observation_week_utc': week_iso,
                'major_change': major_chg,
                'out_of_range': out_of_range,
                'successor_coordinate': succ_coord,
                'successor_safe_status': succ_safe,
                'lock_evidence': lock_ev,
                'manifest_evidence': manifest_ev,
                'unknown_reason': unknown_reason,
            }
            weekly_states.append(state_val)
            weekly_rows_for_unit.append(week_row)
            if week_idx == 12:
                w12_ctx = (prev_valid_proj, prev_obs, prev_state)

        anchor_key = str(rec['anchor_id'])
        baseline_valid_by_anchor[anchor_key] = bool(baseline_valid)
        d90_state: int | None = None
        d90_dt = anchor_dt + dt.timedelta(days=90)
        if baseline_valid and d90_dt <= study_end_dt:
            if 'day_90_snapshot' in rec:
                d90_raw = rec.get('day_90_snapshot')
                if isinstance(d90_raw, dict):
                    d90_snap = _normalize_snapshot(d90_raw)
                    if d90_snap is not None:
                        d90_proj = _select_project(
                            d90_snap,
                            eco,
                            expected_proj,
                            project_path_hint=resolved_path,
                            pom_store=rec.get('pom_store', pom_store),
                            diagnostics=diagnostics,
                        )
                        w12_proj, w12_obs, w12_state = (
                            w12_ctx if w12_ctx is not None else (prev_valid_proj, prev_obs, prev_state)
                        )
                        d90_state, _, _, _ = _evaluate_snapshot_step(
                            base_proj,
                            w12_proj,
                            w12_obs,
                            w12_state,
                            d90_proj,
                            dep,
                            introduced,
                            fixed,
                            verified_aliases,
                        )
            else:
                d90_state = weekly_states[-1] if weekly_states else None
        day_90_state_by_anchor[anchor_key] = d90_state

        reentry_13w, rf_13w = _compute_reentry_flag(weekly_states)
        endpoint_row = weekly_rows_for_unit[-1]
        endpoint_state = endpoint_row['state']
        obs_row = {col: endpoint_row[col] for col in study_contract.OBSERVATION_COLUMNS}
        obs_row['state'] = endpoint_state
        if first_transition_time_utc is not None:
            obs_row['before_commit'] = first_transition_before_commit
            obs_row['after_commit'] = first_transition_after_commit
            obs_row['event_time_lower_utc'] = (first_transition_lower_utc or anchor_dt).isoformat()
            obs_row['event_time_upper_utc'] = first_transition_time_utc.isoformat()
        obs_rows.append(obs_row)

        obs_key_tuple = tuple(obs_row[k] for k in study_contract.OBSERVATION_KEYS)
        stat_bucket = week_stats_by_key.setdefault(obs_key_tuple, {'observed': 0, 'unknown': 0, 'censored': 0})
        seen_unit_weeks: set[Any] = set()
        for wrow in weekly_rows_for_unit:
            w_st = wrow.get('state')
            w_status = str(wrow.get('observation_status') or '')
            if w_st is not None:
                stat_bucket['observed'] += 1
            elif w_status == 'censored':
                stat_bucket['censored'] += 1
            else:
                stat_bucket['unknown'] += 1
            if state_writer is not None:
                study_contract._require_columns(wrow, study_contract.STATE_COLUMNS, 'state_table', streamed_state_count)
                study_contract._validate_state_and_eligibility(wrow, 'state_table', streamed_state_count)
                if w_st is None and not str(wrow.get('unknown_reason') or '').strip():
                    raise ValueError(f'state_table[{streamed_state_count}] state=None requires non-empty unknown_reason')
                wk_val = wrow['observation_week_utc']
                if wk_val in seen_unit_weeks:
                    raise ValueError(f'state_table duplicate weekly observation key: {obs_key_tuple + (wk_val,)!r}')
                seen_unit_weeks.add(wk_val)
                state_writer.writerow(_format_csv_row(study_contract.STATE_COLUMNS, wrow))
                streamed_state_count += 1
            else:
                state_rows.append(wrow)

                                                                                           
        horizon_180_states = list(weekly_states)
        horizon_180_times: list[float] = [float(w * 7) for w in range(1, 14)]
        horizon_180_statuses = [row['observation_status'] for row in weekly_rows_for_unit]
        horizon_180_reasons = [row['unknown_reason'] for row in weekly_rows_for_unit]

        ext_items: list[tuple[float, dict[str, Any]]] = []
        for day_key, raw_ext in extended_raw.items():
            day_val = float(day_key)
            if 91.0 < day_val <= 180.0 and isinstance(raw_ext, dict):
                ext_items.append((day_val, raw_ext))
        ext_items.sort(key=lambda item: item[0])

        for day_val, raw_ext in ext_items:
            ext_dt = anchor_dt + dt.timedelta(days=day_val)
            if ext_dt > study_end_dt:
                horizon_180_states.append(None)
                horizon_180_times.append(day_val)
                horizon_180_statuses.append('censored')
                horizon_180_reasons.append('post_study_endpoint_administrative_censoring')
                continue
            ext_snap = _normalize_snapshot(raw_ext)
            if not baseline_valid or ext_snap is None:
                horizon_180_states.append(None)
                horizon_180_times.append(day_val)
                horizon_180_statuses.append('unknown')
                horizon_180_reasons.append(
                    baseline_fail_reason
                    if not baseline_valid
                    else str(raw_ext.get('missing_reason') or 'missing_extended_git_snapshot')
                )
                prev_valid_proj = None
                prev_obs = None
                prev_state = None
                continue
            ext_proj = _select_project(
                ext_snap,
                eco,
                expected_proj,
                project_path_hint=resolved_path,
                pom_store=rec.get('pom_store', pom_store),
                diagnostics=diagnostics,
            )
            st_val, new_obs, _, reason = _evaluate_snapshot_step(
                base_proj,
                prev_valid_proj,
                prev_obs,
                prev_state,
                ext_proj,
                dep,
                introduced,
                fixed,
                verified_aliases,
            )
            horizon_180_states.append(st_val)
            horizon_180_times.append(day_val)
            if st_val is None:
                horizon_180_statuses.append('unknown')
                horizon_180_reasons.append(reason or 'unknown_extended_state')
                prev_valid_proj = None
                prev_obs = None
                prev_state = None
            else:
                horizon_180_statuses.append('observed')
                horizon_180_reasons.append('')
                prev_valid_proj = ext_proj
                prev_obs = new_obs
                prev_state = st_val

                                                                                                                  
        reentry_180, rf_180 = _compute_reentry_flag(horizon_180_states)
        first_cause = rf_180['first_response_state']
        first_event_time: float | None = None
        censor_time: float | None = None
        censor_reason = ''
        if first_cause is not None:
            for st_val, t_day in zip(horizon_180_states, horizon_180_times):
                if st_val == first_cause:
                    first_event_time = float(t_day)
                    break
        else:
            last_known_day = 0.0
            first_failure_reason = ''
            for st_val, t_day, st_status, st_reason in zip(
                horizon_180_states,
                horizon_180_times,
                horizon_180_statuses,
                horizon_180_reasons,
            ):
                if st_val is not None:
                    last_known_day = float(t_day)
                elif not first_failure_reason:
                    first_failure_reason = st_reason or st_status
            if not baseline_valid:
                censor_time = 0.0
                censor_reason = baseline_fail_reason
            elif anchor_dt + dt.timedelta(days=180) > study_end_dt:
                days_available = max(0.0, (study_end_dt - anchor_dt).total_seconds() / 86400.0)
                censor_time = round(min(180.0, last_known_day if last_known_day > 0 else days_available), 6)
                censor_reason = 'administrative_censoring_before_180d'
            elif first_failure_reason:
                censor_time = round(last_known_day, 6)
                censor_reason = first_failure_reason
            else:
                censor_time = 180.0 if any(t >= 180.0 for t in horizon_180_times) else round(max(horizon_180_times), 6)
                censor_reason = 'administrative_end_of_180d_window_no_event'

        first_row = {col: obs_row[col] for col in study_contract.OBSERVATION_COLUMNS}
        first_row.update({
            'first_event_time': first_event_time,
            'first_cause': first_cause,
            'censor_time': censor_time,
            'censor_reason': censor_reason,
            'reentry_flag': bool(reentry_180 or reentry_13w),
        })
        first_event_rows.append(first_row)

        lineage_rows.append({
            'repository_id': obs_row['repository_id'],
            'project_path': obs_row['project_path'],
            'dependency_coordinate': obs_row['dependency_coordinate'],
            'vulnerability_interval_id': obs_row['vulnerability_interval_id'],
            'anchor_id': obs_row['anchor_id'],
            'ecosystem': obs_row['ecosystem'],
            'family_id': obs_row['family_id'],
            'baseline_commit': obs_row['baseline_commit'],
            'endpoint_commit': obs_row['after_commit'],
            'file_path': obs_row['file_path'],
            'blob_sha256': obs_row['blob_sha256'],
            'source_snapshot_sha256': obs_row['source_snapshot_sha256'],
            'parser_version': obs_row['parser_version'],
        })

        if obs_row['observation_status'] in {'unknown', 'censored'}:
            failure_rows.append({
                'repository_id': obs_row['repository_id'],
                'project_path': obs_row['project_path'],
                'dependency_coordinate': obs_row['dependency_coordinate'],
                'vulnerability_interval_id': obs_row['vulnerability_interval_id'],
                'anchor_id': obs_row['anchor_id'],
                'ecosystem': obs_row['ecosystem'],
                'stage': 'baseline' if not baseline_valid else 'followup_13w',
                'failure_reason': endpoint_row['unknown_reason'] or baseline_fail_reason or obs_row['observation_status'],
                'observation_status': obs_row['observation_status'],
            })

    obs_check = study_contract.validate_observation_table(obs_rows)
    if state_writer is not None:
        state_check = {'row_count': streamed_state_count, 'unique_week_keys': streamed_state_count, 'valid': True}
    else:
        state_check = study_contract.validate_state_table(state_rows)
    first_check = study_contract.validate_first_event_table(first_event_rows)

    status_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    for row in obs_rows:
        st_name = str(row['observation_status'])
        status_counts[st_name] = status_counts.get(st_name, 0) + 1
        s_key = 'null' if row['state'] is None else f"state_{row['state']}"
        state_counts[s_key] = state_counts.get(s_key, 0) + 1

    active_count = sum(1 for r in obs_rows if r['state'] in {1, 2, 4})
    passive_count = sum(1 for r in obs_rows if r['state'] == 3)
    elig_count = sum(1 for r in obs_rows if r['eligibility_only'] is True)
    reentry_count = sum(1 for r in first_event_rows if r['reentry_flag'] is True)

    reconciliation = {
        'study_id': study_contract.STUDY_ID,
        'parser_version': hm.PARSER_VERSION,
        'study_start_utc': STUDY_START_UTC.isoformat(),
        'study_end_utc': study_end_dt.isoformat(),
        'total_observation_units': len(obs_rows),
        'fix_year_summary': {
            'pre_2021_fix_units': pre_2021_fix_count,
            'in_window_2021_plus_fix_units': in_window_fix_count,
            'fix_year_by_signal_year': dict(sorted(fix_signal_cross.items())),
        },
        'horizon_denominators': horizon_counts,
        'endpoint_13w_breakdown': {
            'observation_status_counts': status_counts,
            'state_counts': state_counts,
            'active_git_actions_1_2_4': active_count,
            'passive_lock_resolutions_3': passive_count,
            'eligibility_only_units': elig_count,
            'reentry_units': reentry_count,
        },
        'validations': {
            'observation_table': obs_check,
            'state_table': state_check,
            'first_event_table': first_check,
        },
    }

    return {
        'observation_rows': obs_rows,
        'state_rows': state_rows,
        'first_event_rows': first_event_rows,
        'lineage_rows': lineage_rows,
        'failure_rows': failure_rows,
        'denominator_reconciliation': reconciliation,
        'baseline_valid_by_anchor': baseline_valid_by_anchor,
        'day_90_state_by_anchor': day_90_state_by_anchor,
        'week_stats_by_key': week_stats_by_key,
    }


def _make_npm_snapshot(
    oid: str,
    epoch: float,
    pkg_spec: str = '^1.0.0',
    lock_ver: str | None = '1.0.0',
    present: bool = True,
    corrupt: bool = False,
    pkg_name: str = 'consumer-app',
) -> dict[str, Any]:
    files: dict[str, bytes] = {}
    if corrupt:
        files['package.json'] = b'{invalid json'
    else:
        deps = {'vuln-pkg': pkg_spec} if present else {}
        files['package.json'] = json.dumps({'name': pkg_name, 'dependencies': deps}).encode('utf-8')
        if lock_ver is not None and present:
            files['package-lock.json'] = json.dumps({
                'lockfileVersion': 3,
                'packages': {
                    '': {'name': pkg_name, 'dependencies': deps},
                    'node_modules/vuln-pkg': {'version': lock_ver},
                },
            }).encode('utf-8')
    return {
        'commit_oid': oid,
        'commit_epoch': epoch,
        'files': files,
        'provenance': {p: {'content_sha256': study_contract.sha256_bytes(b)} for p, b in files.items()},
    }


def _make_pypi_snapshot(
    oid: str,
    epoch: float,
    spec: str = '>=1.0.0,<2.0.0',
    lock_ver: str | None = '1.0.0',
    pkg_name: str = 'py-consumer',
) -> dict[str, Any]:
    toml_txt = f'[project]\nname = "{pkg_name}"\ndependencies = ["vuln-py{spec}"]\n'
    files: dict[str, bytes] = {'pyproject.toml': toml_txt.encode('utf-8')}
    if lock_ver is not None:
        uv_txt = f'[[package]]\nname = "vuln-py"\nversion = "{lock_ver}"\n'
        files['uv.lock'] = uv_txt.encode('utf-8')
    return {
        'commit_oid': oid,
        'commit_epoch': epoch,
        'files': files,
        'provenance': {p: {'content_sha256': study_contract.sha256_bytes(b)} for p, b in files.items()},
    }


def _make_maven_snapshot(
    oid: str,
    epoch: float,
    ver: str = '[1.0.0]',
    artifact: str = 'mvn-consumer',
) -> dict[str, Any]:
    pom = (
        f'<project><groupId>org.example</groupId><artifactId>{artifact}</artifactId><version>1.0</version>'
        f'<dependencies><dependency><groupId>org.vuln</groupId><artifactId>vuln-lib</artifactId>'
        f'<version>{ver}</version></dependency></dependencies></project>'
    )
    files: dict[str, bytes] = {'pom.xml': pom.encode('utf-8')}
    return {
        'commit_oid': oid,
        'commit_epoch': epoch,
        'files': files,
        'provenance': {p: {'content_sha256': study_contract.sha256_bytes(b)} for p, b in files.items()},
    }


def generate_synthetic_d1_dataset(seed: int = study_contract.MASTER_SEED) -> list[dict[str, Any]]:
    pass
    _ = study_contract.derive_child_seed(study_contract.STUDY_ID, 'D1', 'synthetic_panel', int(seed))
    base_anchor = dt.datetime(2023, 6, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
    records: list[dict[str, Any]] = []

    def oid_hex(tag: str) -> str:
        return hashlib.sha1(f'{seed}:{tag}'.encode('utf-8')).hexdigest()

    def build_weeks(
        tag: str,
        anchor_dt: dt.datetime,
        fn,
    ) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
        weekly: dict[int, dict[str, Any]] = {}
        ext: dict[int, dict[str, Any]] = {}
        for w in range(1, 14):
            w_dt = anchor_dt + dt.timedelta(weeks=w)
            res = fn(w, w_dt.timestamp())
            if res is not None:
                weekly[w] = res
        for d in (120, 150, 180):
            d_dt = anchor_dt + dt.timedelta(days=d)
            res = fn(d, d_dt.timestamp())
            if res is not None:
                ext[d] = res
        return weekly, ext

                                                
    a1 = base_anchor
    w1, e1 = build_weeks(
        'u1',
        a1,
        lambda step, ep: (
            _make_npm_snapshot(oid_hex('u1_b'), ep, '^1.0.0', '1.0.0')
            if step in (1, 2)
            else _make_npm_snapshot(oid_hex('u1_f'), ep, '^1.2.0', '1.2.0')
        ),
    )
    records.append({
        'repository_id': 'syn_repo_01',
        'project_path': 'package.json',
        'expected_project': 'consumer-app',
        'dependency_coordinate': 'vuln-pkg',
        'vulnerability_interval_id': 'interval_01',
        'anchor_id': 'anc_01',
        'ecosystem': 'npm',
        'family_id': 'fam_01',
        'cve_id': 'CVE-2023-0001',
        'fix_dyad_id': 'dyad_01',
        'introduced_version': '0',
        'fixed_version': '1.2.0',
        'fix_released_at': (a1 - dt.timedelta(days=5)).isoformat(),
        'signal_time_utc': a1.isoformat(),
        'baseline_snapshot': _make_npm_snapshot(oid_hex('u1_b'), (a1 - dt.timedelta(days=1)).timestamp(), '^1.0.0', '1.0.0'),
        'weekly_snapshots': w1,
        'extended_snapshots': e1,
    })

                                                                   
    a2 = base_anchor + dt.timedelta(days=10)
    w2, e2 = build_weeks(
        'u2',
        a2,
        lambda step, ep: (
            _make_npm_snapshot(oid_hex('u2_b'), ep, '^1.0.0', '1.0.0')
            if step == 1
            else _make_npm_snapshot(oid_hex('u2_f'), ep, '^2.0.0', '2.0.0')
        ),
    )
    records.append({
        'repository_id': 'syn_repo_02',
        'project_path': 'package.json',
        'expected_project': 'consumer-app',
        'dependency_coordinate': 'vuln-pkg',
        'vulnerability_interval_id': 'interval_02',
        'anchor_id': 'anc_02',
        'ecosystem': 'npm',
        'family_id': 'fam_02',
        'cve_id': 'CVE-2023-0002',
        'fix_dyad_id': 'dyad_02',
        'introduced_version': '0',
        'fixed_version': '2.0.0',
        'fix_released_at': (a2 - dt.timedelta(days=2)).isoformat(),
        'signal_time_utc': a2.isoformat(),
        'baseline_snapshot': _make_npm_snapshot(oid_hex('u2_b'), (a2 - dt.timedelta(days=1)).timestamp(), '^1.0.0', '1.0.0'),
        'weekly_snapshots': w2,
        'extended_snapshots': e2,
    })

                                                                     
    a3 = base_anchor + dt.timedelta(days=20)
    w3, e3 = build_weeks(
        'u3',
        a3,
        lambda step, ep: (
            _make_pypi_snapshot(oid_hex('u3_b'), ep, '>=1.0.0,<2.0.0', '1.0.0')
            if step <= 3
            else _make_pypi_snapshot(oid_hex('u3_p'), ep, '>=1.0.0,<2.0.0', '1.2.0')
        ),
    )
    records.append({
        'repository_id': 'syn_repo_03',
        'project_path': 'pyproject.toml',
        'expected_project': 'py-consumer',
        'dependency_coordinate': 'vuln-py',
        'vulnerability_interval_id': 'interval_03',
        'anchor_id': 'anc_03',
        'ecosystem': 'PyPI',
        'family_id': 'fam_03',
        'cve_id': 'CVE-2023-0003',
        'fix_dyad_id': 'dyad_03',
        'introduced_version': '0',
        'fixed_version': '1.2.0',
        'fix_released_at': (a3 - dt.timedelta(days=14)).isoformat(),
        'signal_time_utc': a3.isoformat(),
        'baseline_snapshot': _make_pypi_snapshot(oid_hex('u3_b'), (a3 - dt.timedelta(days=1)).timestamp(), '>=1.0.0,<2.0.0', '1.0.0'),
        'weekly_snapshots': w3,
        'extended_snapshots': e3,
    })

                                                                     
    a4 = base_anchor + dt.timedelta(days=30)
    c4_before = oid_hex('u4_b')
    c4_removed = oid_hex('u4_rem')
    w4, e4 = build_weeks(
        'u4',
        a4,
        lambda step, ep: (
            _make_npm_snapshot(c4_before, ep, '^1.0.0', '1.0.0', present=True)
            if step <= 4
            else _make_npm_snapshot(c4_removed, ep, '^1.0.0', None, present=False)
        ),
    )
    records.append({
        'repository_id': 'syn_repo_04',
        'project_path': 'package.json',
        'expected_project': 'consumer-app',
        'dependency_coordinate': 'vuln-pkg',
        'vulnerability_interval_id': 'interval_04',
        'anchor_id': 'anc_04',
        'ecosystem': 'npm',
        'family_id': 'fam_04',
        'cve_id': 'CVE-2023-0004',
        'fix_dyad_id': 'dyad_04',
        'introduced_version': '0',
        'fixed_version': '1.2.0',
        'fix_released_at': (a4 - dt.timedelta(days=3)).isoformat(),
        'signal_time_utc': a4.isoformat(),
        'baseline_snapshot': _make_npm_snapshot(c4_before, (a4 - dt.timedelta(days=1)).timestamp(), '^1.0.0', '1.0.0', present=True),
        'weekly_snapshots': w4,
        'extended_snapshots': e4,
        'verified_aliases': [{
            'old': 'vuln-pkg',
            'verified_no_successor': True,
            'evidence_id': 'EV-REM-04',
            'before_commit': c4_before,
            'after_commit': c4_removed,
        }],
    })

                                                                                 
    a5 = base_anchor + dt.timedelta(days=40)
    w5, e5 = build_weeks(
        'u5',
        a5,
        lambda step, ep: (
            _make_maven_snapshot(oid_hex('u5_fix'), ep, '[1.2.0]')
            if 2 <= step <= 5
            else _make_maven_snapshot(oid_hex('u5_rev' if step >= 6 else 'u5_b'), ep, '[1.0.0]')
        ),
    )
    records.append({
        'repository_id': 'syn_repo_05',
        'project_path': 'pom.xml',
        'expected_project': 'org.example:mvn-consumer',
        'dependency_coordinate': 'org.vuln:vuln-lib',
        'vulnerability_interval_id': 'interval_05',
        'anchor_id': 'anc_05',
        'ecosystem': 'Maven',
        'family_id': 'fam_05',
        'cve_id': 'CVE-2023-0005',
        'fix_dyad_id': 'dyad_05',
        'introduced_version': '0',
        'fixed_version': '1.2.0',
        'fix_released_at': dt.datetime(2020, 8, 15, 0, 0, 0, tzinfo=dt.timezone.utc).isoformat(),
        'signal_time_utc': a5.isoformat(),
        'baseline_snapshot': _make_maven_snapshot(oid_hex('u5_b'), (a5 - dt.timedelta(days=1)).timestamp(), '[1.0.0]'),
        'weekly_snapshots': w5,
        'extended_snapshots': e5,
    })

                                                                                                        
    a6 = base_anchor + dt.timedelta(days=50)
    w6, e6 = build_weeks(
        'u6',
        a6,
        lambda step, ep: _make_npm_snapshot(oid_hex('u6_b'), ep, '^1.0.0', '1.0.0'),
    )
    records.append({
        'repository_id': 'syn_repo_06',
        'project_path': 'package.json',
        'expected_project': 'consumer-app',
        'dependency_coordinate': 'vuln-pkg',
        'vulnerability_interval_id': 'interval_06',
        'anchor_id': 'anc_06',
        'ecosystem': 'npm',
        'family_id': 'fam_06',
        'cve_id': 'CVE-2023-0006',
        'fix_dyad_id': 'dyad_06',
        'introduced_version': '0',
        'fixed_version': '1.2.0',
        'fix_released_at': (a6 - dt.timedelta(days=20)).isoformat(),
        'signal_time_utc': a6.isoformat(),
        'baseline_snapshot': _make_npm_snapshot(oid_hex('u6_b'), (a6 - dt.timedelta(days=1)).timestamp(), '^1.0.0', '1.0.0'),
        'weekly_snapshots': w6,
        'extended_snapshots': e6,
    })

                                                                                 
    a7 = base_anchor + dt.timedelta(days=60)
    w7, e7 = build_weeks(
        'u7',
        a7,
        lambda step, ep: (
            _make_npm_snapshot(oid_hex('u7_b'), ep, '^1.0.0', '1.0.0')
            if step <= 3
            else {'missing': True, 'missing_reason': 'repository_deleted_or_unreachable'}
        ),
    )
    records.append({
        'repository_id': 'syn_repo_07',
        'project_path': 'package.json',
        'expected_project': 'consumer-app',
        'dependency_coordinate': 'vuln-pkg',
        'vulnerability_interval_id': 'interval_07',
        'anchor_id': 'anc_07',
        'ecosystem': 'npm',
        'family_id': 'fam_07',
        'cve_id': 'CVE-2023-0007',
        'fix_dyad_id': 'dyad_07',
        'introduced_version': '0',
        'fixed_version': '1.2.0',
        'fix_released_at': (a7 - dt.timedelta(days=10)).isoformat(),
        'signal_time_utc': a7.isoformat(),
        'baseline_snapshot': _make_npm_snapshot(oid_hex('u7_b'), (a7 - dt.timedelta(days=1)).timestamp(), '^1.0.0', '1.0.0'),
        'weekly_snapshots': w7,
        'extended_snapshots': e7,
    })

                                                                                     
    a8 = dt.datetime(2026, 6, 15, 0, 0, 0, tzinfo=dt.timezone.utc)
    w8, e8 = build_weeks(
        'u8',
        a8,
        lambda step, ep: _make_pypi_snapshot(oid_hex('u8_b'), ep, '==1.0.0', '1.0.0'),
    )
    records.append({
        'repository_id': 'syn_repo_08',
        'project_path': 'pyproject.toml',
        'expected_project': 'py-consumer',
        'dependency_coordinate': 'vuln-py',
        'vulnerability_interval_id': 'interval_08',
        'anchor_id': 'anc_08',
        'ecosystem': 'PyPI',
        'family_id': 'fam_08',
        'cve_id': 'CVE-2026-0008',
        'fix_dyad_id': 'dyad_08',
        'introduced_version': '0',
        'fixed_version': '1.2.0',
        'fix_released_at': (a8 - dt.timedelta(days=1)).isoformat(),
        'signal_time_utc': a8.isoformat(),
        'baseline_snapshot': _make_pypi_snapshot(oid_hex('u8_b'), (a8 - dt.timedelta(days=1)).timestamp(), '==1.0.0', '1.0.0'),
        'weekly_snapshots': w8,
        'extended_snapshots': e8,
    })

    return records


def _write_optional_parquet(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> bool:
    pass
    try:
        import pandas as pd
        clean_rows = []
        for row in rows:
            item = {}
            for col in columns:
                val = row.get(col)
                item[col] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list, tuple)) else val
            clean_rows.append(item)
        df = pd.DataFrame(clean_rows, columns=list(columns))
        tmp = path.with_suffix(path.suffix + '.tmp')
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        return True
    except Exception:
        return False


def run_d1_panel(
    records: Iterable[dict[str, Any]] | None,
    out_dir: Path | str,
    mode: str = 'synthetic',
    seed: int = study_contract.MASTER_SEED,
    study_end_utc: str | dt.datetime = STUDY_END_UTC,
    input_paths: list[Path | str] | tuple[Path | str, ...] = (),
    drop_bulk_rows: bool = False,
) -> dict[str, Any]:
    pass
    if mode not in {'official', 'synthetic'}:
        raise ValueError(f'Unsupported D1 execution mode: {mode!r}')
    if mode == 'official':
        validation.assert_official_empirical_unlocked()
        if records is None:
            raise ValueError('Official D1 execution requires explicit records input')
    else:
        if input_paths:
            validation.assert_synthetic_mode_isolated(input_paths)
        if records is None:
            records = generate_synthetic_d1_dataset(seed=seed)

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    obs_csv = target_dir / 'git_observations.csv'
    state_csv = target_dir / 'weekly_states.csv'
    first_csv = target_dir / 'first_exit.csv'
    lineage_csv = target_dir / 'lineage.csv'
    failures_csv = target_dir / 'acquisition_failures.csv'
    recon_json = target_dir / 'denominator_reconciliation.json'

    stream_states = (mode == 'official') or bool(drop_bulk_rows)
    panel = build_panel_tables(
        records,
        study_end_utc=study_end_utc,
        stream_state_csv_path=state_csv if stream_states else None,
    )

    study_contract.atomic_write_csv(obs_csv, study_contract.OBSERVATION_COLUMNS, panel['observation_rows'])
    if not stream_states:
        study_contract.atomic_write_csv(state_csv, study_contract.STATE_COLUMNS, panel['state_rows'])
    study_contract.atomic_write_csv(first_csv, study_contract.FIRST_EVENT_COLUMNS, panel['first_event_rows'])
    study_contract.atomic_write_csv(lineage_csv, LINEAGE_COLUMNS, panel['lineage_rows'])
    if drop_bulk_rows:
        panel['lineage_rows'] = []
    study_contract.atomic_write_csv(failures_csv, FAILURE_COLUMNS, panel['failure_rows'])
    if drop_bulk_rows:
        panel['failure_rows'] = []

    parquet_written = {
        'git_observations.parquet': _write_optional_parquet(
            target_dir / 'git_observations.parquet',
            study_contract.OBSERVATION_COLUMNS,
            panel['observation_rows'],
        ),
        'first_exit.parquet': _write_optional_parquet(
            target_dir / 'first_exit.parquet',
            study_contract.FIRST_EVENT_COLUMNS,
            panel['first_event_rows'],
        ),
    }
    if drop_bulk_rows:
        panel['first_event_rows'] = []

    reconciliation = dict(panel['denominator_reconciliation'])
    reconciliation['mode'] = mode
    reconciliation['recorded_at'] = study_contract.utc_now_iso()
    reconciliation['parquet_emitted'] = parquet_written
    study_contract.atomic_write_json(recon_json, reconciliation)

    return {
        'mode': mode,
        'out_dir': str(target_dir),
        'artifacts': {
            'git_observations_csv': str(obs_csv),
            'weekly_states_csv': str(state_csv),
            'first_exit_csv': str(first_csv),
            'lineage_csv': str(lineage_csv),
            'acquisition_failures_csv': str(failures_csv),
            'denominator_reconciliation_json': str(recon_json),
        },
        'observation_rows': panel['observation_rows'],
        'state_rows': panel['state_rows'],
        'first_event_rows': panel['first_event_rows'],
        'week_stats_by_key': panel['week_stats_by_key'],
        'denominator_reconciliation': reconciliation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='D1 immutable Git panel builder.')
    parser.add_argument('-m', choices=['synthetic', 'official'], default='synthetic', help='Execution mode')
    parser.add_argument('-o', required=True, help='Output directory path')
    parser.add_argument('-s', type=int, default=study_contract.MASTER_SEED, help='Deterministic seed')
    parser.add_argument('-w', type=int, default=DEFAULT_DURABLE_WORKERS, help='Collection worker count cap (1..8)')
    parser.add_argument('-i', default=None, help='Optional JSON records file')
    args = parser.parse_args(argv)

    if args.w < 1 or args.w > MAX_DURABLE_WORKERS:
        raise ValueError(f'Worker cap violated: {args.w} > {MAX_DURABLE_WORKERS}')

    if args.m == 'official':
        validation.assert_official_empirical_unlocked()

    records = None
    in_paths: tuple[Path, ...] = ()
    if args.i:
        in_path = Path(args.i).resolve()
        in_paths = (in_path,)
        if args.m == 'synthetic':
            validation.assert_synthetic_mode_isolated(in_paths)
        records = json.loads(in_path.read_text(encoding='utf-8'))

    result = run_d1_panel(
        records=records,
        out_dir=Path(args.o),
        mode=args.m,
        seed=args.s,
        input_paths=in_paths,
    )
    print(json.dumps({
        'mode': result['mode'],
        'out_dir': result['out_dir'],
        'total_units': result['denominator_reconciliation']['total_observation_units'],
        'artifacts': result['artifacts'],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
