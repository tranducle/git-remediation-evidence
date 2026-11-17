"""Parse ecosystem-specific dependency and version evidence conservatively."""
from functools import total_ordering
import re
from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet, InvalidSpecifier


@total_ordering
class SemVer:
    def __init__(self, text):
        match = re.fullmatch(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?', str(text))
        if not match:
            raise ValueError('Not an exact SemVer version')
        self.release = tuple(int(x) for x in match.groups()[:3])
        self.pre = tuple(match[4].split('.')) if match[4] is not None else ()
        if any(not x or (x.isdigit() and len(x) > 1 and x.startswith('0')) for x in self.pre):
            raise ValueError('Invalid SemVer prerelease identifier')
        if match[5] is not None and any(not x for x in match[5].split('.')):
            raise ValueError('Invalid SemVer build identifier')

    @property
    def major(self):
        return self.release[0]

    def __eq__(self, other):
        return isinstance(other, SemVer) and (self.release, self.pre) == (other.release, other.pre)

    def __lt__(self, other):
        if self.release != other.release:
            return self.release < other.release
        if not self.pre or not other.pre:
            return bool(self.pre) and not other.pre
        for left, right in zip(self.pre, other.pre):
            if left == right:
                continue
            if left.isdigit() and right.isdigit():
                return int(left) < int(right)
            if left.isdigit() != right.isdigit():
                return left.isdigit()
            return left < right
        return len(self.pre) < len(other.pre)


def ordered(value, ecosystem):
    try:
        if value is None:
            return None
        if ecosystem == 'npm':
            return SemVer(str(value))
        if ecosystem == 'PyPI':
            return Version(str(value))
        if ecosystem == 'Maven' and re.fullmatch(r'\d+(?:\.\d+)*', str(value)):
                                                                           
                                                                                 
            parts = [int(x) for x in str(value).split('.')]
            while len(parts) > 1 and parts[-1] == 0:
                parts.pop()
            return tuple(parts + [0] * (16 - len(parts))) if len(parts) <= 16 else None
    except (InvalidVersion, ValueError):
        pass
    return None


def exact_pin(spec, ecosystem):
    value = str(spec or '').strip()
    if ecosystem == 'PyPI':
        if not value.startswith('==') or value.startswith('===') or ',' in value or '*' in value:
            return None
        value = value[2:].strip()
    elif ecosystem == 'npm':
        value = value.removeprefix('=')
    elif ecosystem == 'Maven' and re.fullmatch(r'\[[^,\[\]]+\]', value):
        value = value[1:-1]
    return value if ordered(value, ecosystem) is not None else None


def affected(pin, introduced, fixed, ecosystem):
    candidate, high = ordered(pin, ecosystem), ordered(fixed, ecosystem)
    low = None if str(introduced) == '0' else ordered(introduced, ecosystem)
    if candidate is None or high is None or (low is None and str(introduced) != '0'):
        return None
    if low is not None and low >= high:
        return None
    return (low is None or candidate >= low) and candidate < high


def major_change(old, new, ecosystem):
    left, right = ordered(old, ecosystem), ordered(new, ecosystem)
    if left is None or right is None:
        return None
    if ecosystem == 'Maven':
        return right[0] != left[0]
    return right.major != left.major


def _npm_branches(spec):
    pass
    result = []
    for branch in str(spec).split('||'):
        branch = branch.strip()
        if branch in {'', '*', 'x', 'X'}:
            result.append([])
            continue
        tokens = re.findall(r'(>=|<=|>|<|=|\^|~)?\s*([0-9A-Za-z.+*-]+)', branch)
        consumed = re.sub(r'\s+', '', ''.join(op + value for op, value in tokens))
        if consumed != re.sub(r'\s+', '', branch):
            return None
        comparisons = []
        for op, value in tokens:
            try:
                if op in {'^', '~'}:
                    fields = value.split('.')
                    if any(not x.isdigit() for x in fields) or not 1 <= len(fields) <= 3:
                        return None
                    parts = [int(x) for x in fields] + [0] * (3 - len(fields))
                    if op == '~':
                        index = 0 if len(fields) == 1 else 1
                    elif parts[0] > 0 or len(fields) == 1:
                        index = 0
                    elif parts[1] > 0 or len(fields) == 2:
                        index = 1
                    else:
                        index = 2
                    upper = parts[:]
                    upper[index] += 1
                    upper[index + 1:] = [0] * (2 - index)
                    comparisons += [('>=', SemVer('.'.join(map(str, parts)))),
                                    ('<', SemVer('.'.join(map(str, upper))))]
                elif any(x in value for x in '*xX') or (not op and re.fullmatch(r'\d+(?:\.\d+)?', value)):
                    if op:
                        return None
                    fields = value.split('.')
                    prefix = []
                    wildcard = False
                    for field in fields:
                        if field in {'*', 'x', 'X'}:
                            wildcard = True
                        elif field.isdigit() and not wildcard:
                            prefix.append(int(field))
                        else:
                            return None
                    if not prefix:
                        continue
                    if len(prefix) >= 3:
                        return None
                    lo = prefix + [0] * (3 - len(prefix))
                    hi = lo[:]
                    hi[len(prefix) - 1] += 1
                    hi[len(prefix):] = [0] * (3 - len(prefix))
                    comparisons += [('>=', SemVer('.'.join(map(str, lo)))),
                                    ('<', SemVer('.'.join(map(str, hi))))]
                else:
                    comparisons.append((op or '=', SemVer(value)))
            except ValueError:
                return None
        result.append(comparisons)
    return result


def allows(spec, pin, ecosystem):
    candidate = ordered(pin, ecosystem)
    if candidate is None:
        return None
    if ecosystem == 'PyPI':
        try:
            return SpecifierSet(str(spec)).contains(str(pin), prereleases=True)
        except InvalidSpecifier:
            return None
    if ecosystem == 'Maven':
        exact = exact_pin(spec, ecosystem)
        return ordered(exact, ecosystem) == candidate if exact else None
    branches = _npm_branches(spec)
    if branches is None:
        return None
    def matches(comparisons):
        if candidate.pre and not any(v.pre and v.release == candidate.release for _, v in comparisons):
            return False
        return all({'=': candidate == target, '>=': candidate >= target, '<=': candidate <= target,
                    '>': candidate > target, '<': candidate < target}[op] for op, target in comparisons)
    return any(matches(x) for x in branches)


def enforces_fixed(spec, fixed, ecosystem):
    target = ordered(fixed, ecosystem)
    if target is None:
        return None
    pin = exact_pin(spec, ecosystem)
    if pin is not None:
        return ordered(pin, ecosystem) >= target
    if ecosystem == 'Maven':
        return None
    if ecosystem == 'npm':
        branches = _npm_branches(spec)
        if branches is None:
            return None
        return all(any(op in {'=', '>', '>='} and bound >= target for op, bound in row) for row in branches)
    try:
        clauses = SpecifierSet(str(spec))
    except InvalidSpecifier:
        return None
    for clause in clauses:
        if clause.operator in {'>=', '>', '~=', '=='} and '*' not in clause.version:
            bound = ordered(clause.version, ecosystem)
            if bound is not None and bound >= target:
                return True
    return False
