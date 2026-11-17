"""Parse ecosystem-specific dependency and version evidence conservatively."""
from __future__ import annotations
import ast
import configparser
import csv
import io
import json
import posixpath
from pathlib import PurePosixPath
import re
import tomllib

from packaging.markers import default_environment
from packaging.requirements import Requirement, InvalidRequirement
import yaml

from maven_models import effective_model, xml, text
from construct_rules import wholly_affected, constraint_major
from version_rules import ordered, exact_pin, affected, allows, enforces_fixed, major_change

PARSER_VERSION = 'git-measurement-v4.0'
LOCKS = {'package-lock.json', 'npm-shrinkwrap.json', 'yarn.lock', 'pnpm-lock.yaml',
         'poetry.lock', 'uv.lock', 'Pipfile.lock'}


class SourceYamlLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        return _unique_pairs([(self.construct_object(k, deep=deep), self.construct_object(v, deep=deep))
                              for k,v in node.value])


def normalize(name, ecosystem):
    return re.sub(r'[-_.]+', '-', name).lower() if ecosystem == 'PyPI' else name


def decoded(data):
    return data.decode('utf-8-sig', errors='strict')


def _unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate source key: ' + str(key))
        result[key] = value
    return result


def strict_json(data):
    result = json.loads(decoded(data), object_pairs_hook=_unique_pairs)
    if not isinstance(result, dict):
        raise ValueError('Expected a JSON object')
    return result


def project(snapshot, path, name, ecosystem):
    return dict(project_path=path, project_name=name, ecosystem=ecosystem, dependencies={},
                locked_versions={}, dependency_set_complete=True, uncertain_dependencies=[],
                notes=[], lock_conflicts=[], source_objects={path: snapshot.get('provenance', {}).get(path, {})},
                dependency_scopes={}, dependency_declarations={}, manifest_keys={}, parser_version=PARSER_VERSION,
                commit_oid=snapshot['commit_oid'], commit_epoch=snapshot.get('commit_epoch'),
                bot_config_paths=snapshot.get('bot_config_paths', []))


def ancestor_files(snapshot, path, basename):
    for parent in PurePosixPath(path).parents:
        candidate = str(parent / basename)
        if candidate in snapshot['files']:
            yield candidate, snapshot['files'][candidate]


def bind(p, snapshot, path):
    p['source_objects'][path] = snapshot.get('provenance', {}).get(path, {})


def _node_location_candidates(project_dir, alias):
    directory = PurePosixPath(project_dir)
    for parent in [directory, *directory.parents]:
        yield posixpath.normpath(str(parent / 'node_modules' / alias)).removeprefix('./')


def _npm_lock(snapshot, manifest_path, lock_path, data, aliases, specs):
    lock = strict_json(data)
    root = PurePosixPath(lock_path).parent
    directory = str(PurePosixPath(manifest_path).parent.relative_to(root))
    found = {}
    if 'packages' in lock:
        packages = lock['packages']
        importer = packages.get('' if directory == '.' else directory)
        if not isinstance(importer, dict):
            return {}, ['npm lock has no matching project/importer metadata']
        declared = {**importer.get('devDependencies', {}), **importer.get('dependencies', {}), **importer.get('optionalDependencies', {})}
        for actual, keys in aliases.items():
            pins = set()
            for alias in keys:
                if declared.get(alias) != specs[alias]:
                    continue
                for location in _node_location_candidates(directory, alias):
                    row = packages.get(location)
                    if row is not None:
                        if isinstance(row, dict) and not row.get('link') and ordered(row.get('version'), 'npm') is not None:
                            if row.get('name', actual) == actual:
                                pins.add(row['version'])
                        break
            if len(pins) == 1:
                found[actual] = next(iter(pins))
    elif lock.get('lockfileVersion') in [None, 1] and directory == '.':
        for actual, keys in aliases.items():
            pins = {lock.get('dependencies', {}).get(key, {}).get('version') for key in keys}
            pins = {x for x in pins if ordered(x, 'npm') is not None}
            if len(pins) == 1:
                found[actual] = next(iter(pins))
    else:
        return {}, ['Unsupported npm lock schema or v1 workspace linkage']
    return found, []


def _yarn_lock(data, aliases, specs):
    entries = []
    descriptors, pin = [], None
    for line in decoded(data).splitlines() + ['end-of-source:']:
        if line and not line[0].isspace() and not line.startswith('#'):
            if descriptors and pin:
                entries.append((descriptors, pin))
            header = line.removesuffix(':').strip()
            descriptors = [x.strip().strip('"\'') for x in next(csv.reader([header], skipinitialspace=True))]
            pin = None
        else:
            match = re.fullmatch(r'\s+version(?::|\s)\s*["\']?([^"\'\s]+)["\']?\s*', line)
            if match:
                pin = match[1]
    found = {}
    for actual, keys in aliases.items():
        wanted = {key + '@' + specs[key] for key in keys} | {key + '@npm:' + specs[key] for key in keys}
        pins = {pin for descriptors, pin in entries if wanted.intersection(descriptors) and ordered(pin, 'npm') is not None}
        if len(pins) == 1:
            found[actual] = next(iter(pins))
    return found, []


def _pnpm_lock(manifest_path, lock_path, data, aliases, specs):
    lock = yaml.load(decoded(data), Loader=SourceYamlLoader)
    if not isinstance(lock, dict):
        return {}, ['pnpm lock is not a mapping']
    importer = str(PurePosixPath(manifest_path).parent.relative_to(PurePosixPath(lock_path).parent))
    selected = lock.get('importers', {}).get(importer) if 'importers' in lock else lock if importer == '.' else None
    if not isinstance(selected, dict):
        return {}, ['pnpm matching importer is absent']
    dependencies = {**selected.get('devDependencies', {}), **selected.get('dependencies', {}), **selected.get('optionalDependencies', {})}
    found = {}
    for actual, keys in aliases.items():
        pins = set()
        for key in keys:
            row = dependencies.get(key)
            if isinstance(row, dict):
                if row.get('specifier') != specs[key]:
                    continue
                pin = row.get('version')
            else:
                if selected.get('specifiers', {}).get(key) != specs[key]:
                    continue
                pin = row
            if isinstance(pin, str):
                pin = pin.split('(', 1)[0]
                                                                                  
                if pin.startswith(actual + '@'):
                    pin = pin[len(actual) + 1:]
                if ordered(pin, 'npm') is not None:
                    pins.add(pin)
        if len(pins) == 1:
            found[actual] = next(iter(pins))
    return found, []


def npm_project(snapshot, path):
    manifest = strict_json(snapshot['files'][path])
    p = project(snapshot, path, manifest.get('name'), 'npm')
    specs = {**manifest.get('devDependencies', {}), **manifest.get('dependencies', {}), **manifest.get('optionalDependencies', {})}
    aliases = {}
    for key, spec in specs.items():
        if not isinstance(spec, str):
            p['dependency_set_complete'] = False
            p['notes'].append('Nonstring npm dependency specifier')
            continue
        actual, constraint = key, spec
        if spec.startswith('npm:'):
            alias = re.fullmatch(r'npm:((?:@[^/]+/)?[^@]+)@(.+)', spec)
            if alias is None:
                p['uncertain_dependencies'].append(key)
                continue
            actual, constraint = alias.groups()
        elif re.match(r'^(?:file:|link:|workspace:|git[+:]|https?:|ssh:|github:|gitlab:|bitbucket:|\.{0,2}/)', spec):
            p['uncertain_dependencies'].append(key)
        aliases.setdefault(actual, []).append(key)
        if actual in p['dependencies'] and p['dependencies'][actual] != constraint:
            p['uncertain_dependencies'].append(actual)
        p['dependencies'][actual] = constraint
        scopes = [scope for section, scope in [('dependencies', 'default'), ('optionalDependencies', 'optional'),
                   ('devDependencies', 'development')] if key in manifest.get(section, {})]
        p['dependency_scopes'][actual] = scopes[0]
        for section, scope in [('dependencies', 'default'), ('optionalDependencies', 'optional'), ('devDependencies', 'development')]:
            if key in manifest.get(section, {}):
                declared = manifest[section][key]
                if isinstance(declared, str) and declared.startswith('npm:'):
                    match = re.fullmatch(r'npm:((?:@[^/]+/)?[^@]+)@(.+)', declared)
                    declared = match[2] if match else declared
                p['dependency_declarations'].setdefault(actual, []).append(dict(constraint=declared, scope=scope, path=path))
    p['manifest_keys'] = aliases
    candidates = []
    manager = manifest.get('packageManager', '').split('@', 1)[0]
    selected = {'npm': ['npm-shrinkwrap.json', 'package-lock.json'], 'yarn': ['yarn.lock'],
                'pnpm': ['pnpm-lock.yaml']}.get(manager, ['npm-shrinkwrap.json', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml'])
    for basename in selected:
        sources = list(ancestor_files(snapshot, path, basename))
        if not sources:
            continue
        lock_path, data = sources[0]
                                                                            
        if basename == 'package-lock.json' and any(PurePosixPath(x[0]).parent == PurePosixPath(lock_path).parent
                                                 for x in ancestor_files(snapshot, path, 'npm-shrinkwrap.json')):
            continue
        bind(p, snapshot, lock_path)
        if basename in {'package-lock.json', 'npm-shrinkwrap.json'}:
            pins, notes = _npm_lock(snapshot, path, lock_path, data, aliases, specs)
        elif basename == 'yarn.lock':
            pins, notes = _yarn_lock(data, aliases, specs)
        else:
            pins, notes = _pnpm_lock(path, lock_path, data, aliases, specs)
        p['notes'] += notes
        candidates.append((lock_path, pins))
    for package in p['dependencies']:
        versions = {pins[package] for _, pins in candidates if package in pins}
        if len(versions) == 1:
            p['locked_versions'][package] = next(iter(versions))
        elif len(versions) > 1:
            p['lock_conflicts'].append(package)
            p['notes'].append('Conflicting recorded lock versions: ' + package)
    return p


def requirement(values, p, marker_environment=None):
    for value in values:
        try:
            req = Requirement(value.strip())
        except InvalidRequirement:
            if value.strip():
                p['dependency_set_complete'] = False
                p['notes'].append('Unparsed Python dependency declaration')
            continue
        name = normalize(req.name, 'PyPI')
        if req.marker:
            variables = set(re.findall(r'\b[A-Za-z_][A-Za-z_0-9]*\b', str(req.marker))) & (set(default_environment()) | {'extra'})
            if marker_environment is None or not variables.issubset(marker_environment):
                p['uncertain_dependencies'].append(name)
                p['notes'].append('Conditional Python dependency needs recorded environment: ' + name)
                continue
            if not req.marker.evaluate(environment=marker_environment):
                continue
        if req.url:
            p['uncertain_dependencies'].append(name)
            p['notes'].append('Direct-URL Python version is not inferred: ' + name)
            continue
        constraint = str(req.specifier)
        if name in p['dependencies'] and p['dependencies'][name] != constraint:
            constraint = ','.join([p['dependencies'][name], constraint])
        p['dependencies'][name] = constraint


def scoped_requirements(values, p, scope, marker_environment=None, source_path=None):
    temporary = dict(p, dependencies={}, uncertain_dependencies=[], notes=[])
    requirement(values, temporary, marker_environment)
    p['dependency_set_complete'] &= temporary['dependency_set_complete']
    p['uncertain_dependencies'] += temporary['uncertain_dependencies']
    p['notes'] += temporary['notes']
    for name, constraint in temporary['dependencies'].items():
        p['dependencies'].setdefault(name, constraint)
        p['dependency_scopes'].setdefault(name, scope)
        p['dependency_declarations'].setdefault(name, []).append(
            dict(constraint=constraint, scope=scope, path=source_path or p['project_path']))


def poetry_constraint(value):
    if not isinstance(value, str):
        return None
    if value[:1] not in {'^', '~'} or value.startswith('~='):
        return value if value.startswith(('=', '<', '>', '!', '~')) else '==' + value if re.fullmatch(r'\d+(?:\.\d+)*', value) else None
    fields = value[1:].split('.')
    if any(not x.isdigit() for x in fields):
        return None
    numbers = [int(x) for x in fields]
    if value[0] == '^':
        index = next((i for i, x in enumerate(numbers) if x), len(numbers) - 1)
    else:
        index = 0 if len(numbers) == 1 else len(numbers) - 2
    hi = numbers[:index + 1]
    hi[index] += 1
    return '>=' + '.'.join(fields) + ',<' + '.'.join(map(str, hi))


def python_project(snapshot, path, marker_environment=None):
    basename = PurePosixPath(path).name
    p = project(snapshot, path, None, 'PyPI')
    if basename == 'pyproject.toml':
        data = tomllib.loads(decoded(snapshot['files'][path]))
        standard = data.get('project', {})
        poetry = data.get('tool', {}).get('poetry', {})
        p['project_name'] = standard.get('name') or poetry.get('name')
        if 'dependencies' in standard.get('dynamic', []):
            p['dependency_set_complete'] = False
            p['notes'].append('Dynamic PEP-621 dependency metadata is not evaluated')
        if 'dependencies' in standard:
            scoped_requirements(standard['dependencies'], p, 'default', marker_environment)
        for values in standard.get('optional-dependencies', {}).values():
            scoped_requirements(values, p, 'extras', marker_environment)
        if 'dependencies' not in standard and poetry:
            for key, value in poetry.get('dependencies', {}).items():
                if key == 'python':
                    continue
                actual = normalize(key, 'PyPI')
                if isinstance(value, dict) and any(x in value for x in ['markers','python','platform','optional','git','path','url']):
                    p['uncertain_dependencies'].append(actual)
                    continue
                constraint = poetry_constraint(value.get('version') if isinstance(value, dict) else value)
                if constraint is None:
                    p['uncertain_dependencies'].append(actual)
                else:
                    p['dependencies'][actual] = constraint
                    p['dependency_scopes'][actual] = 'default'
                    p['dependency_declarations'].setdefault(actual, []).append(dict(constraint=constraint, scope='default', path=path))
        for group in poetry.get('group', {}).values():
            for key, value in group.get('dependencies', {}).items():
                constraint = poetry_constraint(value)
                if constraint is None:
                    p['uncertain_dependencies'].append(normalize(key, 'PyPI'))
                else:
                    scoped_requirements([key + constraint], p, 'development', marker_environment)
    elif basename == 'setup.cfg':
        cfg = configparser.ConfigParser(interpolation=None)
        cfg.read_string(decoded(snapshot['files'][path]))
        p['project_name'] = cfg.get('metadata', 'name', fallback=None)
        declarations = cfg.get('options', 'install_requires', fallback='')
        if declarations.lstrip().startswith(('file:', 'attr:')):
            p['dependency_set_complete'] = False
            p['notes'].append('Indirect setup.cfg dependency source is not resolved')
        else:
            scoped_requirements(declarations.splitlines(), p, 'default', marker_environment)
        if cfg.has_section('options.extras_require'):
            for _, declarations in cfg.items('options.extras_require'):
                scoped_requirements(declarations.splitlines(), p, 'extras', marker_environment)
    else:
        tree = ast.parse(decoded(snapshot['files'][path]))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                 ((isinstance(n.func, ast.Name) and n.func.id == 'setup') or
                  (isinstance(n.func, ast.Attribute) and n.func.attr == 'setup'))]
        conditional = len(calls) != 1 or any(isinstance(n, (ast.If, ast.IfExp)) for n in ast.walk(tree))
        if conditional:
            p['dependency_set_complete'] = False
            p['notes'].append('Conditional or ambiguous setup call is not executed')
        for call in calls:
            for kw in call.keywords:
                if kw.arg is None:
                    p['dependency_set_complete'] = False
                    p['notes'].append('Dynamic setup keyword expansion is not executed')
                elif kw.arg in {'name','install_requires','extras_require'}:
                    try:
                        value = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        p['dependency_set_complete'] = False
                        p['notes'].append('Dynamic setup metadata is not executed')
                        continue
                    if kw.arg == 'name' and isinstance(value, str):
                        p['project_name'] = value
                    elif kw.arg == 'install_requires' and isinstance(value, list):
                        scoped_requirements(value, p, 'default', marker_environment)
                    elif kw.arg == 'extras_require' and isinstance(value, dict):
                        for declarations in value.values():
                            if not isinstance(declarations, list):
                                p['dependency_set_complete'] = False
                            else:
                                scoped_requirements(declarations, p, 'extras', marker_environment)
        if conditional:
            p['uncertain_dependencies'] += list(p['dependencies'])
    companion = str(PurePosixPath(path).parent / 'requirements.txt')
    if companion in snapshot['files']:
        requirements = parse_requirements_file(snapshot, companion, marker_environment)
        p['dependency_set_complete'] &= requirements['dependency_set_complete']
        p['uncertain_dependencies'] += requirements['uncertain_dependencies']
        p['source_objects'].update(requirements['source_objects'])
        for name, constraint in requirements['dependencies'].items():
                                                                               
                                                                              
            p['dependencies'].setdefault(name, constraint)
            p['dependency_scopes'].setdefault(name, 'default')
            p['dependency_declarations'].setdefault(name, []).append(dict(constraint=constraint, scope='default', path=companion))
    pins = {}
    for basename in ['poetry.lock', 'uv.lock', 'Pipfile.lock']:
        lock_path = str(PurePosixPath(path).parent / basename)
        if lock_path not in snapshot['files']:
            continue
        bind(p, snapshot, lock_path)
        if basename == 'Pipfile.lock':
            lock = strict_json(snapshot['files'][lock_path])
            rows = {**lock.get('develop', {}), **lock.get('default', {})}
            rows = [dict(name=k, version=exact_pin(v.get('version'), 'PyPI'), marker=v.get('markers')) for k,v in rows.items()]
        else:
            rows = tomllib.loads(decoded(snapshot['files'][lock_path])).get('package', [])
        for row in rows:
            name = normalize(str(row.get('name', '')), 'PyPI')
            if name not in p['dependencies']:
                continue
            if row.get('marker') or row.get('markers'):
                p['uncertain_dependencies'].append(name)
            if ordered(row.get('version'), 'PyPI') is not None:
                pins.setdefault(name, set()).add(str(row['version']))
    for name, versions in pins.items():
        if len(versions) == 1:
            p['locked_versions'][name] = next(iter(versions))
        else:
            p['lock_conflicts'].append(name)
    return p


def parse_requirements_file(snapshot, path, marker_environment=None, stack=()):
    if path in stack or len(stack) > 16 or path not in snapshot['files']:
        raise ValueError('Unresolved or cyclic requirements include')
    p = project(snapshot, path, None, 'PyPI')
    contents = decoded(snapshot['files'][path]).replace('\\\n', '')
    for line in contents.splitlines():
        line = re.split(r'\s+#', line, maxsplit=1)[0].strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'-r\s+([^\s]+)', line)
        if match:
            child = posixpath.normpath(str(PurePosixPath(path).parent / match[1]))
            if child.startswith('../'):
                raise ValueError('Requirements include escapes snapshot')
            nested = parse_requirements_file(snapshot, child, marker_environment, stack + (path,))
            requirement([name + spec for name,spec in nested['dependencies'].items()], p, marker_environment)
            p['uncertain_dependencies'] += nested['uncertain_dependencies']
            p['dependency_set_complete'] &= nested['dependency_set_complete']
            p['source_objects'].update(nested['source_objects'])
        elif line.startswith('-'):
            p['dependency_set_complete'] = False
            p['notes'].append('Requirements option/constraint needs explicit resolution')
        else:
            requirement([line], p, marker_environment)
    return p


def parse_projects(snapshot, ecosystem, expected_name=None, *, pom_store=None,
                   marker_environment=None, active_profiles=None, diagnostics=None):
    result = []
    priorities = {'pyproject.toml':0, 'setup.cfg':1, 'setup.py':2}
    chosen = {}
    for path in snapshot['files']:
        name = PurePosixPath(path).name
        if name in priorities:
            if name == 'pyproject.toml':
                try:
                    metadata = tomllib.loads(decoded(snapshot['files'][path]))
                except (ValueError, UnicodeError):
                    metadata = {'project': {'unreadable': True}}
                if 'project' not in metadata and not metadata.get('tool', {}).get('poetry'):
                                                                           
                                                                     
                    continue
            directory = str(PurePosixPath(path).parent)
            if directory not in chosen or priorities[name] < priorities[PurePosixPath(chosen[directory]).name]:
                chosen[directory] = path
    for path in sorted(snapshot['files']):
        name = PurePosixPath(path).name
        try:
            if ecosystem == 'npm' and name == 'package.json':
                p = npm_project(snapshot, path)
            elif ecosystem == 'PyPI' and path in chosen.values():
                p = python_project(snapshot, path, marker_environment)
            elif ecosystem == 'Maven' and name == 'pom.xml':
                model = effective_model(snapshot, path, pom_store, active_profiles)
                p = project(snapshot, path, model['group'] + ':' + model['artifact'], ecosystem)
                p.update(dependencies=model['resolved'], dependency_scopes=model['scopes'],
                         dependency_set_complete=model['complete'], notes=model['notes'])
                root = xml(snapshot['files'][path])
                p['managed_only'] = text(root, 'packaging') == 'pom' and not model['dependencies'] and bool(model['management'])
                p['dependency_scopes'] = {key: {'compile':'default','runtime':'default'}.get(value, value)
                                          for key,value in p['dependency_scopes'].items()}
                p['management_evidence'] = {':'.join(key[:2]):value for key,value in model['management'].items()}
                if model['activation_unknown']:
                    p['uncertain_dependencies'] = list(model['resolved'])
                p['source_objects'].update(model['evidence'])
            else:
                continue
            if expected_name is None or (p['project_name'] and normalize(p['project_name'], ecosystem) == normalize(expected_name, ecosystem)):
                result.append(p)
        except (ValueError, TypeError, KeyError, AttributeError, SyntaxError, configparser.Error, yaml.YAMLError) as error:
            if diagnostics is not None:
                diagnostics.append(dict(path=path, error_type=type(error).__name__, error=str(error)))
    return result


def observation(p, package, introduced, fixed):
    key = normalize(package, p['ecosystem'])
    spec = p['dependencies'].get(key)
    row = dict(present=True, version=None, affected=None, constraint=spec, basis='unknown',
               parser_version=PARSER_VERSION)
    row['dependency_scope'] = p['dependency_scopes'].get(key, 'unresolved')
    if p.get('managed_only'):
        row.update(present=None, basis='parent dependencyManagement is not a consuming project unit',
                   dependency_scope='dependency_management')
        return row
    if key in p.get('uncertain_dependencies', []):
        row.update(present=None, basis='conditional or incomplete dependency evidence')
        return row
    if key not in p['dependencies']:
        complete = p.get('dependency_set_complete', False)
        row.update(present=False if complete else None,
                   basis='complete direct dependency set' if complete else 'incomplete direct dependency set')
        return row
    if key in p.get('lock_conflicts', []):
        row['basis'] = 'conflicting lock evidence'
        return row
    locked = p['locked_versions'].get(key)
    declarations = p.get('dependency_declarations', {}).get(key, [])
    constraints = list(dict.fromkeys(x['constraint'] for x in declarations)) or [spec]
    row['declarations'] = declarations
    if locked is None and len(constraints) > 1:
        results = [wholly_affected(x, introduced, fixed, p['ecosystem']) for x in constraints]
        row.update(affected=True if all(x is True for x in results) else None,
                   basis='separate direct-scope alternatives; no installed version inferred')
        return row
    pin = locked or exact_pin(spec, p['ecosystem'])
    if locked is not None:
        consistent = all(allows(x, pin, p['ecosystem']) is True for x in constraints)
        if consistent is not True:
            row.update(version=pin, basis='manifest and lock conflict' if consistent is False else 'constraint grammar is unsupported')
            return row
    row.update(version=pin, affected=affected(pin, introduced, fixed, p['ecosystem']),
               basis='recorded lock version' if locked else 'exact recorded declaration' if pin else 'range without recorded resolved version')
    if pin is None and wholly_affected(spec, introduced, fixed, p['ecosystem']) is True:
        row.update(affected=True, basis='every permitted declared version is affected; resolution unobserved')
    return row


def transition(before, after, package, introduced, fixed, verified_aliases=None):
    old, new = observation(before, package, introduced, fixed), observation(after, package, introduced, fixed)
    row = dict(state=None, before=old, after=new, commit_oid=after['commit_oid'], reason='unclassified')
    if old['affected'] is False and new['affected'] is True:
        row.update(state=0, reason='recorded reversion restores current exposure; first response is not erased')
        return row
    if old['affected'] is not True:
        row['reason'] = 'baseline affected version is unverified'
        return row
    if before['commit_oid'] == after['commit_oid']:
        row.update(state=0, reason='same immutable downstream source')
        return row
    if new['present'] is None:
        row['reason'] = 'post-event dependency evidence incomplete'
        return row
    if new['present'] is False:
        added = set(after['dependencies']) - set(before['dependencies'])
        aliases = [x for x in (verified_aliases or []) if x.get('old') == package and x.get('new') in added]
        if aliases:
            if len(aliases) != 1:
                row['reason'] = 'ambiguous successor evidence'
                return row
            alias = aliases[0]
            if not all(alias.get(x) for x in ['verified','evidence_id','comparable','successor_fixed']):
                row['reason'] = 'successor compatibility is unverified'
                return row
            if alias.get('before_commit') != before['commit_oid'] or alias.get('after_commit') != after['commit_oid']:
                row['reason'] = 'successor evidence is not bound to this commit pair'
                return row
            successor = observation(after, alias['new'], '0', alias['successor_fixed'])
            major = major_change(old['version'], successor['version'], after['ecosystem'])
            if successor['affected'] is not False or major is None:
                row['reason'] = 'successor patch or major-line status unknown'
                return row
            row.update(state=2 if major else 1, reason='verified same-commit safe successor')
            return row
                                                                              
                                                                                  
        if added:
            row['reason'] = 'new coordinates require successor review'
            return row
        removal = [x for x in (verified_aliases or []) if x.get('old') == package and
                   x.get('verified_no_successor') is True and x.get('evidence_id') and
                   x.get('before_commit') == before['commit_oid'] and x.get('after_commit') == after['commit_oid']]
        if removal:
            row.update(state=4, reason='commit-bound direct removal and no-safe-successor audit')
        else:
            row.update(candidate_state=4, reason='direct removal candidate requires successor/retained-use audit', removal_audit_required=True)
        return row
    if new['affected'] is True:
        row.update(state=0, reason='recorded version remains affected')
        return row
    if new['affected'] is None:
        row['reason'] = 'post-event version or consistency is unknown'
        return row
    if new['constraint'] == old['constraint'] and new.get('declarations') == old.get('declarations'):
        if new['version'] != old['version'] and normalize(package, after['ecosystem']) in after['locked_versions']:
            row.update(state=3, reason='recorded lock update without manifest edit')
        else:
            row['reason'] = 'no evidenced downstream resolution change'
        return row
    if enforces_fixed(new['constraint'], fixed, after['ecosystem']) is not True:
        row['reason'] = 'manifest edit does not establish a fixed-version requirement'
        return row
    major = major_change(old['version'], new['version'], after['ecosystem'])
    if major is None:
        old_major = constraint_major(old['constraint'], before['ecosystem'])
        new_major = constraint_major(new['constraint'], after['ecosystem'])
        if old_major is not None and new_major is not None:
            major = old_major != new_major
    if major is None:
        row['reason'] = 'major-line comparison is unsupported'
    else:
        row.update(state=2 if major else 1, reason='recorded fixed manifest update', major_change=major)
    return row
