"""Test conservative parsing of immutable package metadata."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

measurement = importlib.import_module(os.environ.get('MEASUREMENT_MODULE', 'history_measurement'))


def snapshot(oid, files):
    raw = {path: value.encode() for path,value in files.items()}
    return dict(commit_oid=oid, commit_epoch=1_700_000_000, files=raw,
                provenance={k:dict(content_sha256=hashlib.sha256(v).hexdigest()) for k,v in raw.items()},
                bot_config_paths=[])


def npm(oid, spec='^1.0.0', pin='1.0.0', extra=None):
    dependencies = {'risk': spec} if spec is not None else {}
    dependencies.update(extra or {})
    files = {'package.json':json.dumps(dict(name='consumer',dependencies=dependencies)),
             'package-lock.json':json.dumps(dict(lockfileVersion=3,packages={
                 '':dict(name='consumer',dependencies=dependencies),
                 'node_modules/risk':dict(version=pin)}))}
    return measurement.parse_projects(snapshot(oid,files),'npm','consumer')[0]


class ReproducedDefects(unittest.TestCase):
    def test_npm_workspace_uses_nearest_scoped_lock_version(self):
        files = {'packages/consumer/package.json':'{"name":"consumer","dependencies":{"risk":"^1.0.0"}}',
                 'package-lock.json':json.dumps(dict(lockfileVersion=3,packages={
                     'packages/consumer':dict(dependencies={'risk':'^1.0.0'}),
                     'packages/consumer/node_modules/risk':dict(version='1.0.1'),
                     'node_modules/risk':dict(version='1.2.0')}))}
        p=measurement.parse_projects(snapshot('a',files),'npm','consumer')[0]
        self.assertEqual(measurement.observation(p,'risk','0','1.1.0')['version'],'1.0.1')

    def test_python_marker_is_not_discarded(self):
        p=measurement.parse_projects(snapshot('a',{'pyproject.toml':'[project]\nname="consumer"\ndependencies=["risk==1.0.0; sys_platform == \'win32\'"]\n'}),'PyPI','consumer')[0]
        self.assertIsNone(measurement.observation(p,'risk','0','1.1.0')['affected'])

    def test_dynamic_pep621_is_not_absent_dependency(self):
        p=measurement.parse_projects(snapshot('a',{'pyproject.toml':'[project]\nname="consumer"\ndynamic=["dependencies"]\n'}),'PyPI','consumer')[0]
        self.assertIsNone(measurement.observation(p,'risk','0','1.1.0')['present'])

    def test_ambiguous_python_lock_does_not_choose_last_version(self):
        p=measurement.parse_projects(snapshot('a',{'pyproject.toml':'[project]\nname="consumer"\ndependencies=["risk>=1.0"]\n',
             'uv.lock':'[[package]]\nname="risk"\nversion="1.0.0"\n[[package]]\nname="risk"\nversion="1.2.0"\n'}),'PyPI','consumer')[0]
        self.assertIsNone(measurement.observation(p,'risk','0','1.1.0')['affected'])

    def test_zero_major_minor_increment_is_not_a_major_change(self):
        r=measurement.transition(npm('a','^0.1.0','0.1.0'),npm('b','^0.2.0','0.2.0'),'risk','0','0.2.0')
        self.assertEqual(r['state'],1)

    def test_local_parent_management_is_evidenced(self):
        files={'pom.xml':'<project><groupId>g</groupId><artifactId>parent</artifactId><version>1</version><dependencyManagement><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId><version>1.0.0</version></dependency></dependencies></dependencyManagement></project>',
               'child/pom.xml':'<project><parent><groupId>g</groupId><artifactId>parent</artifactId><version>1</version></parent><artifactId>child</artifactId><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId></dependency></dependencies></project>'}
        p=measurement.parse_projects(snapshot('a',files),'Maven','g:child')[0]
        self.assertTrue(measurement.observation(p,'x:risk','0','1.1.0')['affected'])
        self.assertIn('pom.xml',p['source_objects'])


class Invariants(unittest.TestCase):
    def test_recorded_reversion_restores_current_state_zero(self):
        self.assertEqual(measurement.transition(npm('a','^1.1.0','1.1.0'),npm('b','^1.0.0','1.0.0'),'risk','0','1.1.0')['state'],0)

    def test_unsupported_constraint_cannot_validate_a_passive_event(self):
        a=npm('a','^1.0.0 garbage','1.0.0');b=npm('b','^1.0.0 garbage','1.1.0')
        self.assertIsNone(measurement.observation(a,'risk','0','1.1.0')['affected'])
        self.assertIsNone(measurement.transition(a,b,'risk','0','1.1.0')['state'])

    def test_release_eligibility_does_not_create_action(self):
        self.assertEqual(measurement.transition(npm('a'),npm('b'),'risk','0','1.1.0')['state'],0)

    def test_passive_requires_recorded_lock_change(self):
        self.assertEqual(measurement.transition(npm('a'),npm('b',pin='1.1.0'),'risk','0','1.1.0')['state'],3)

    def test_active_fixed_manifest_change(self):
        self.assertEqual(measurement.transition(npm('a'),npm('b','^1.1.0','1.1.0'),'risk','0','1.1.0')['state'],1)

    def test_manifest_lock_conflict_is_unknown(self):
        p=npm('a','^2.0.0','1.0.0')
        self.assertIsNone(measurement.observation(p,'risk','0','1.1.0')['affected'])

    def test_fixed_lock_with_unfixed_manifest_floor_is_not_active(self):
        self.assertIsNone(measurement.transition(npm('a'),npm('b','^1.0.1','1.1.0'),'risk','0','1.1.0')['state'])

    def test_importer_constraint_mismatch_is_unknown(self):
        files={'packages/consumer/package.json':'{"name":"consumer","dependencies":{"risk":"^1.0.0"}}',
               'pnpm-lock.yaml':'importers:\n  packages/consumer:\n    dependencies:\n      risk:\n        specifier: ^2.0.0\n        version: 2.0.1\n'}
        p=measurement.parse_projects(snapshot('a',files),'npm','consumer')[0]
        self.assertIsNone(measurement.observation(p,'risk','0','1.1.0')['affected'])

    def test_npm_packages_schema_does_not_fall_back_to_legacy_section(self):
        files={'package.json':'{"name":"consumer","dependencies":{"risk":"^1.0.0"}}',
               'package-lock.json':'{"lockfileVersion":3,"packages":{"":{"dependencies":{"risk":"^1.0.0"}}},"dependencies":{"risk":{"version":"1.0.0"}}}'}
        p=measurement.parse_projects(snapshot('a',files),'npm','consumer')[0]
        self.assertIsNone(measurement.observation(p,'risk','0','1.1.0')['affected'])

    def test_unknown_parent_cannot_establish_removal(self):
        xml='<project><parent><groupId>g</groupId><artifactId>parent</artifactId><version>1</version><relativePath/></parent><artifactId>consumer</artifactId></project>'
        p=measurement.parse_projects(snapshot('a',{'pom.xml':xml}),'Maven','g:consumer')[0]
        self.assertIsNone(measurement.observation(p,'x:risk','0','1.1.0')['present'])

    def test_maven_profile_is_not_implicitly_host_activated(self):
        xml='<project><groupId>g</groupId><artifactId>consumer</artifactId><profiles><profile><id>conditional</id><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId><version>1.0.0</version></dependency></dependencies></profile></profiles></project>'
        p=measurement.parse_projects(snapshot('a',{'pom.xml':xml}),'Maven','g:consumer')[0]
        self.assertIsNone(measurement.observation(p,'x:risk','0','1.1.0')['present'])

    def test_python_url_is_not_empty_unconditional_specifier(self):
        text='[project]\nname="consumer"\ndependencies=["risk @ https://example.invalid/risk.zip"]\n'
        p=measurement.parse_projects(snapshot('a',{'pyproject.toml':text}),'PyPI','consumer')[0]
        self.assertIsNone(measurement.observation(p,'risk','0','1.1.0')['present'])

    def test_same_commit_is_not_an_action(self):
        self.assertEqual(measurement.transition(npm('a'),npm('a'),'risk','0','1.1.0')['state'],0)


class AdditionalFormats(unittest.TestCase):
    def test_build_only_pyproject_preserves_static_setup_cfg_metadata(self):
        files={'pyproject.toml':'[build-system]\nbuild-backend="setuptools.build_meta"\n',
               'setup.cfg':'[metadata]\nname = consumer\n[options]\ninstall_requires =\n    risk==1.0.0\n'}
        p=measurement.parse_projects(snapshot('a',files),'PyPI','consumer')[0]
        self.assertEqual(p['project_path'],'setup.cfg')
        self.assertTrue(measurement.observation(p,'risk','0','1.1.0')['affected'])

    def test_python_explicit_marker_environment(self):
        text='[project]\nname="consumer"\ndependencies=["risk==1.0.0; sys_platform == \'win32\'"]\n'
        p=measurement.parse_projects(snapshot('a',{'pyproject.toml':text}),'PyPI','consumer',marker_environment={'sys_platform':'linux'})[0]
        self.assertFalse(measurement.observation(p,'risk','0','1.1.0')['present'])

    def test_requirements_includes_are_snapshot_bound(self):
        files={'requirements.txt':'-r base.txt\n','base.txt':'risk==1.0.0\n'}
        p=measurement.parse_requirements_file(snapshot('a',files),'requirements.txt')
        self.assertTrue(measurement.observation(p,'risk','0','1.1.0')['affected'])
        self.assertIn('base.txt',p['source_objects'])

    def test_requirements_cycles_are_rejected(self):
        with self.assertRaises(ValueError):
            measurement.parse_requirements_file(snapshot('a',{'a.txt':'-r a.txt\n'}),'a.txt')

    def test_yarn_descriptor_matches_direct_constraint(self):
        files={'package.json':'{"name":"consumer","dependencies":{"risk":"^1.0.0"},"packageManager":"yarn@1.22.0"}',
               'yarn.lock':'"risk@^1.0.0":\n  version "1.0.1"\n\n"risk@^2.0.0":\n  version "2.0.1"\n'}
        p=measurement.parse_projects(snapshot('a',files),'npm','consumer')[0]
        self.assertEqual(measurement.observation(p,'risk','0','1.1.0')['version'],'1.0.1')

    def test_poetry_caret_is_not_a_pep440_exact_pin(self):
        text='[tool.poetry]\nname="consumer"\n[tool.poetry.dependencies]\npython="^3.11"\nrisk="^1.0.0"\n'
        p=measurement.parse_projects(snapshot('a',{'pyproject.toml':text}),'PyPI','consumer')[0]
        self.assertEqual(p['dependencies']['risk'],'>=1.0.0,<2')
        self.assertIsNone(measurement.observation(p,'risk','0','1.1.0')['affected'])


if __name__ == '__main__':
    unittest.main()
