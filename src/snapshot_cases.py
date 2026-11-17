"""Classify a sequence of immutable snapshot cases."""
import json
from pathlib import Path
from history_measurement import parse_projects, observation, transition

def parser_case(case_path):
    case = json.loads(case_path.read_text())
    projects, diagnostics = [], []
    for index, row in enumerate(case['snapshots']):
        files = {entry['source_path']: (case_path.parent / entry['path']).read_bytes() for entry in row['files']}
        snapshot = dict(commit_oid=row['snapshot_id'], commit_epoch=None, files=files,
                        provenance={entry['source_path']:dict(content_sha256=entry['sha256']) for entry in row['files']},
                        bot_config_paths=[])
        parsed = parse_projects(snapshot, case['ecosystem'], case['expected_project'], diagnostics=diagnostics)
        projects.append(parsed[0] if len(parsed) == 1 else None)
    baseline = observation(projects[0],case['dependency'],case['introduced'],case['fixed']) if projects[0] else None
    status = 'unknown'
    if baseline and baseline['affected'] is True:
        status = 'affected'
    elif baseline and (baseline['affected'] is False or baseline['present'] is False):
        status = 'not_affected'
    events, first, state = [], None, 0 if status == 'affected' else None
    for before, after in zip(projects,projects[1:]):
        if status != 'affected' or before is None or after is None:
            state = None
            events.append(dict(state=None,reason='unverified baseline or project identity'))
            continue
        result = transition(before,after,case['dependency'],case['introduced'],case['fixed'])
        events.append(result)
        current = result['state']
        old_obs, new_obs = result['before'], result['after']
        if current is None and old_obs['affected'] is False and new_obs['affected'] is False:
                                                                               
                                               
            continue
        state = current
        if first is None and current in {1,2,3,4}:
            first = current
    if not case['source_window_complete']:
        state = None
    return dict(case_id=case['case_id'], project_path=projects[0]['project_path'] if projects[0] else None,
                baseline_status=status, baseline_observation=baseline, endpoint_state=state,
                first_response_state=first, events=events, parse_diagnostics=diagnostics,
                dependency_scope=baseline.get('dependency_scope', 'unresolved') if baseline else 'unresolved',
                scope='Exact project identity; all recorded direct scopes retained and stratified; no parent/child aggregation')
