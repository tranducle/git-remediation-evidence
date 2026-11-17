"""Parse ecosystem-specific dependency and version evidence conservatively."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import posixpath
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET


def text(element, name):
    found = element.find(name)
    return found.text.strip() if found is not None and found.text else ''


def xml(data):
    if len(data) > 8_000_000 or any(x in data.upper() for x in [b'<!DOCTYPE', b'<!ENTITY']):
        raise ValueError('Unsafe or oversized POM')
    root = ET.fromstring(data)
    for node in root.iter():
        node.tag = node.tag.split('}')[-1]
    if root.tag != 'project' or sum(1 for _ in root.iter()) > 30_000:
        raise ValueError('Unexpected POM root or node count')
    return root


def expand(value, properties):
    for _ in range(32):
        newer = re.sub(r'\$\{([^}]+)\}', lambda m: properties.get(m[1], m[0]), value)
        if newer == value:
            break
        value = newer
    return value


def depkey(element):
    return (text(element, 'groupId'), text(element, 'artifactId'),
            text(element, 'type') or 'jar', text(element, 'classifier'))


class PomStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.index = json.loads((self.root / 'index.json').read_text())

    def get(self, coordinate, epoch):
        row = self.index.get(coordinate)
        if not row:
            raise ValueError('Pinned POM unavailable: ' + coordinate)
        path = (self.root / row['path']).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('POM cache path is invalid')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Pinned POM checksum mismatch')
        if 'SNAPSHOT' in coordinate.upper():
            raise ValueError('Mutable snapshot POM has no verified historical artifact identity')
        date = row.get('available_from')
        if not date or epoch is None:
            raise ValueError('POM historical availability is unverified')
        available = dt.datetime.fromisoformat(date.replace('Z', '+00:00'))
        if available.tzinfo is None or available.timestamp() > epoch:
            raise ValueError('POM was not evidenced available before this commit')
        return data, dict(row, coordinate=coordinate)


def effective_model(snapshot, path, store=None, active_profiles=None, stack=()):
    if len(stack) >= 24 or path in stack:
        raise ValueError('Parent/import POM cycle or depth limit')
    root = xml(snapshot['files'][path])
    notes, evidence = [], {}
    complete = True
    activation_unknown = False
    own_props = {p.tag: str(p.text or '').strip() for p in root.findall('properties/*')}
    parent_model = dict(properties={}, dependencies={}, management={}, complete=True, evidence={}, activation_unknown=False)
    parent = root.find('parent')
    if parent is not None:
        pg, pa, pv = [expand(text(parent, x), own_props) for x in ['groupId', 'artifactId', 'version']]
        expected = pg + ':' + pa + ':' + pv
        relative_node = parent.find('relativePath')
        relative = '../pom.xml' if relative_node is None else str(relative_node.text or '').strip()
        local_path = posixpath.normpath(str(PurePosixPath(path).parent / relative)) if relative else None
        parent_snapshot, parent_path = snapshot, None
        if local_path and not local_path.startswith('../') and local_path in snapshot['files']:
            candidate = xml(snapshot['files'][local_path])
            local_coord = ':'.join(text(candidate, x) or text(candidate, 'parent/' + x)
                                   for x in ['groupId', 'artifactId', 'version'])
            if local_coord == expected:
                parent_path = local_path
        if parent_path is None:
            try:
                if store is None:
                    raise ValueError('External parent has no pinned source store: ' + expected)
                raw, receipt = store.get(expected, snapshot.get('commit_epoch'))
                candidate = xml(raw)
                actual = ':'.join(text(candidate, x) or text(candidate, 'parent/' + x)
                                  for x in ['groupId', 'artifactId', 'version'])
                if actual != expected:
                    raise ValueError('External parent coordinate mismatch')
                parent_path = '@pom/' + expected
                parent_snapshot = dict(snapshot, files={**snapshot['files'], parent_path: raw})
                evidence[parent_path] = receipt
            except (ValueError, OSError) as error:
                complete = False
                notes.append(str(error))
        if parent_path is not None:
            parent_model = effective_model(parent_snapshot, parent_path, store, active_profiles, stack + (path,))
            complete &= parent_model['complete']
            activation_unknown |= parent_model['activation_unknown']
            notes += parent_model['notes']
            evidence.update(parent_model['evidence'])
            if parent_path in snapshot.get('provenance', {}):
                evidence[parent_path] = snapshot['provenance'][parent_path]
    properties = {**parent_model['properties'], **own_props}
    group = expand(text(root, 'groupId') or text(root, 'parent/groupId'), properties)
    artifact = expand(text(root, 'artifactId'), properties)
    version = expand(text(root, 'version') or text(root, 'parent/version'), properties)
    properties.update({'project.groupId': group, 'pom.groupId': group,
                       'project.artifactId': artifact, 'pom.artifactId': artifact,
                       'project.version': version, 'pom.version': version,
                       'project.parent.version': text(root, 'parent/version')})
    for key in list(properties):
        properties[key] = expand(properties[key], properties)
    selected = [root]
    profiles = root.findall('profiles/profile')
    if profiles and active_profiles is None:
        complete = False
        activation_unknown = True
        notes.append('Profile activation environment is unknown; dependency set may differ')
    elif profiles:
        for profile in profiles:
            if profile.find('activation') is not None:
                complete = False
                activation_unknown = True
                notes.append('Automatic profile activation requires separate environment evidence')
            if text(profile, 'id') in active_profiles:
                selected.append(profile)
                for prop in profile.findall('properties/*'):
                    properties[prop.tag] = expand(str(prop.text or '').strip(), properties)
    management = dict(parent_model['management'])
    dependencies = dict(parent_model['dependencies'])
    own_management = {}
    imports = []
    for container in selected:
        for dep in container.findall('dependencyManagement/dependencies/dependency'):
            key = tuple(expand(x, properties) for x in depkey(dep))
            if text(dep, 'scope') == 'import' and key[2] == 'pom':
                imports.append((key, expand(text(dep, 'version'), properties)))
            else:
                own_management[key] = dict(version=text(dep, 'version'), scope=text(dep, 'scope') or 'compile')
        for dep in container.findall('dependencies/dependency'):
            key = tuple(expand(x, properties) for x in depkey(dep))
            inherited = dependencies.get(key, {})
            dependencies[key] = dict(version=text(dep, 'version') or inherited.get('version', ''), scope=text(dep, 'scope') or inherited.get('scope', ''),
                                     optional=text(dep, 'optional') == 'true')
    management.update(own_management)
    for key, version in imports:
        coordinate = key[0] + ':' + key[1] + ':' + version
        try:
            if store is None:
                raise ValueError('Imported BOM has no pinned source store: ' + coordinate)
            raw, receipt = store.get(coordinate, snapshot.get('commit_epoch'))
            bom_root = xml(raw)
            actual = ':'.join(text(bom_root, x) or text(bom_root, 'parent/' + x)
                              for x in ['groupId', 'artifactId', 'version'])
            if actual != coordinate:
                raise ValueError('Imported BOM coordinate mismatch')
            bom_path = '@pom/' + coordinate
            bom = effective_model(dict(snapshot, files={**snapshot['files'], bom_path: raw}),
                                  bom_path, store, active_profiles, stack + (path,))
            complete &= bom['complete']
            activation_unknown |= bom['activation_unknown']
            notes += bom['notes']
            evidence.update(bom['evidence'])
            evidence[bom_path] = receipt
            for managed_key, row in bom['management'].items():
                                                                              
                                                                
                management.setdefault(managed_key, dict(row, version=expand(row['version'], bom['properties'])))
        except (ValueError, OSError) as error:
            complete = False
            notes.append(str(error))
    resolved, scopes = {}, {}
    for key, row in dependencies.items():
        key = tuple(expand(x, properties) for x in key)
        managed = management.get(key, {})
        scope = expand(row['scope'] or managed.get('scope') or 'compile', properties)
        coordinate = key[0] + ':' + key[1]
        pin = expand(row['version'] or managed.get('version', ''), properties)
        if not pin or '${' in pin or not key[0] or not key[1]:
            pin = ''
            notes.append('Unresolved version or coordinate: ' + coordinate)
        if coordinate in resolved and resolved[coordinate] != pin:
            resolved[coordinate] = ''
            notes.append('Multiple type/classifier versions: ' + coordinate)
        else:
            resolved.setdefault(coordinate, pin)
        scopes[coordinate] = scope
    return dict(properties=properties, dependencies=dependencies, management=management,
                resolved=resolved, scopes=scopes, group=group, artifact=artifact, version=version,
                complete=bool(complete), notes=notes, evidence=evidence, activation_unknown=bool(activation_unknown))
