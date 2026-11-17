"""Parse ecosystem-specific dependency and version evidence conservatively."""
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from version_rules import ordered, exact_pin, affected, allows, _npm_branches

SCOPES = ('default', 'optional', 'development', 'provided', 'test', 'extras',
          'workspace', 'child_module', 'dependency_management',
          'dependency_management_and_child_module', 'build_plugin', 'unresolved')
VERSION_KEYS = ('manifest_ranges', 'exact_pins', 'unversioned_declarations',
                'properties_and_inheritance', 'lockfile_resolutions', 'non_target_observations')


def empty_version_evidence():
    return {key: [] for key in VERSION_KEYS}


def bounds(spec, ecosystem):
    pass
    if ecosystem == 'npm':
        rows = _npm_branches(spec)
        if rows is None or any(v.pre for row in rows for _, v in row):
            return None
    elif ecosystem == 'PyPI':
        try:
            clauses = list(SpecifierSet(spec))
        except InvalidSpecifier:
            return None
        row = []
        for clause in clauses:
            op, version = clause.operator, clause.version
            if op == '!=':
                continue
            if op == '~=':
                v = ordered(version, ecosystem)
                if v is None or len(v.release) < 2:
                    return None
                upper = list(v.release[:-1]); upper[-1] += 1
                row += [('>=', v), ('<', ordered('.'.join(map(str, upper)), ecosystem))]
            elif op in {'>', '>=', '<', '<=', '=='} and '*' not in version:
                v = ordered(version, ecosystem)
                if v is None:
                    return None
                row.append(('=' if op == '==' else op, v))
            else:
                return None
        rows = [row]
    else:
        return None
    result = []
    for row in rows:
        lows = [(v, op != '>') for op, v in row if op in {'=', '>', '>='}]
        highs = [(v, op != '<') for op, v in row if op in {'=', '<', '<='}]
        lo = max(lows, key=lambda x: (x[0], not x[1])) if lows else (None, False)
        hi = min(highs, key=lambda x: (x[0], x[1])) if highs else (None, False)
        if lo[0] is not None and hi[0] is not None:
            if lo[0] > hi[0] or (lo[0] == hi[0] and not (lo[1] and hi[1])):
                return None
        result.append((lo, hi))
    return result


def wholly_affected(spec, introduced, fixed, ecosystem):
    pin = exact_pin(spec, ecosystem)
    if pin is not None:
        return affected(pin, introduced, fixed, ecosystem)
    intervals = bounds(spec, ecosystem)
    high = ordered(fixed, ecosystem)
    low = None if str(introduced) == '0' else ordered(introduced, ecosystem)
    if not intervals or high is None or (low is None and str(introduced) != '0'):
        return None
    if low is not None and low >= high:
        return None
    for (lo, lo_inclusive), (hi, inclusive) in intervals:
        if low is not None and (lo is None or lo < low):
            return None
        if hi is None or hi > high or (hi == high and inclusive):
            return None
                                                                             
                                                                   
        if ecosystem == 'npm':
            release = list(lo.release if lo is not None else (0, 0, 0))
            if lo is not None and not lo_inclusive:
                release[2] += 1
            witness = '.'.join(map(str, release))
            if allows(spec, witness, ecosystem) is not True:
                return None
        else:
            start = lo if lo is not None else ordered('0', ecosystem)
            release = list(start.release) + [0] * max(0, 3 - len(start.release))
            candidates = [str(start), str(start) + '.post1']
            for step in (1, 2, 3):
                tail = release[:]; tail[-1] += step
                candidates.append((str(start.epoch) + '!' if start.epoch else '') + '.'.join(map(str, tail)))
            if not any(allows(spec, value, ecosystem) is True for value in candidates):
                return None
    return True


def constraint_major(spec, ecosystem):
    pin = exact_pin(spec, ecosystem)
    if pin is not None:
        v = ordered(pin, ecosystem)
        return v[0] if ecosystem == 'Maven' else v.major
    intervals = bounds(spec, ecosystem)
    if not intervals:
        return None
    majors = set()
    for (lo, _), (hi, inclusive) in intervals:
        if lo is None or hi is None:
            return None
        top = hi.major
        release = hi.release
        if not inclusive and all(x == 0 for x in release[1:]):
            top -= 1
        if lo.major != top:
            return None
        majors.add(top)
    return next(iter(majors)) if len(majors) == 1 else None


def validate_form(form, completed=False):
    pass
    for row in form['case_results']:
        if row.get('dependency_scope') not in SCOPES and not (not completed and row.get('dependency_scope') is None):
            raise ValueError('dependency_scope is outside the closed enum')
        ve = row.get('version_evidence')
        if not isinstance(ve, dict) or set(ve) != set(VERSION_KEYS) or any(type(v) is not list for v in ve.values()):
            raise ValueError('version_evidence requires exactly six array keys')
        for key in ('endpoint_state', 'first_response_state'):
            state = row.get(key)
            allowed = {0, 1, 2, 3, 4} if key == 'endpoint_state' else {1, 2, 3, 4}
            if state is not None and (type(state) is not int or state not in allowed):
                raise ValueError('States require integers or null')
        if completed and row.get('baseline_status') not in {'affected', 'not_affected', 'unknown'}:
            raise ValueError('Completed baseline status is required')
        if completed and row['baseline_status'] != 'affected' and any(row.get(k) is not None for k in ('endpoint_state', 'first_response_state')):
            raise ValueError('Unestablished affected baseline requires null outcomes')
    return True
