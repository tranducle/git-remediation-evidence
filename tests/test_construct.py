"""Test conservative parsing of immutable package metadata."""
import copy
import json
import unittest
from construct_rules import SCOPES, empty_version_evidence, validate_form, wholly_affected
from test_measurement import snapshot
import history_measurement as m


class ConstructV4(unittest.TestCase):
    def test_range_inside_affected_no_resolution(self):
        p = m.parse_projects(snapshot('a', {'package.json':json.dumps({'name':'p','dependencies':{'risk':'^1.5.3'}})}), 'npm', 'p')[0]
        obs = m.observation(p, 'risk', '0', '3.0.1')
        self.assertIs(obs['affected'], True)
        self.assertIsNone(obs['version'])
        self.assertEqual(m.transition(p, p, 'risk', '0', '3.0.1')['state'], 0)

    def test_straddling_unbounded_and_unsupported_are_unknown(self):
        for spec in ['^1.5.3', '*', '>=1.0.0', '>=1.0.0 <=3.0.1', '>=2.0.0 <1.0.0', 'git:other', '^1.0.0 || ^4.0.0']:
            fixed = '1.9.0' if spec == '^1.5.3' else '3.0.1'
            self.assertIsNone(wholly_affected(spec, '0', fixed, 'npm'), spec)

    def test_introduced_lower_boundary_is_respected(self):
        self.assertIsNone(wholly_affected('^1.0.0', '1.5.0', '3.0.1', 'npm'))
        self.assertTrue(wholly_affected('>=1.5.0 <2.0.0', '1.5.0', '3.0.1', 'npm'))

    def test_pep440_bounds_and_compatible_release(self):
        self.assertTrue(wholly_affected('~=0.19', '0', '1.2.2', 'PyPI'))
        self.assertTrue(wholly_affected('>=1.0,<1.5', '1.0', '1.5', 'PyPI'))
        self.assertIsNone(wholly_affected('>=1.0,<=1.5', '1.0', '1.5', 'PyPI'))
        self.assertIsNone(wholly_affected('>=1.5,<1.0', '0', '2.0', 'PyPI'))
        self.assertIsNone(wholly_affected('>=1,<2,!=1.*', '0', '3', 'PyPI'))
        self.assertIsNone(wholly_affected('>=1,<=1,!=1', '0', '3', 'PyPI'))
        self.assertIsNone(wholly_affected('>1.0.0 <1.0.1', '0', '3.0.0', 'npm'))

    def test_single_snapshot_complete_and_incomplete_windows(self):
        import hashlib
        from pathlib import Path
        import tempfile
        from snapshot_cases import parser_case
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);raw=b'{"name":"p","dependencies":{"risk":"^1.0.0"}}'
            (root/'source.json').write_bytes(raw)
            case=dict(case_id='synthetic',ecosystem='npm',expected_project='p',dependency='risk',introduced='0',fixed='3.0.0',source_window_complete=True,
                snapshots=[dict(snapshot_id='s0',files=[dict(source_path='package.json',path='source.json',sha256=hashlib.sha256(raw).hexdigest())])])
            path=root/'case.json';path.write_text(json.dumps(case))
            self.assertEqual(parser_case(path)['endpoint_state'],0)
            self.assertIsNone(parser_case(path)['first_response_state'])
            case['source_window_complete']=False;path.write_text(json.dumps(case))
            self.assertIsNone(parser_case(path)['endpoint_state'])

    def test_npm_development_scope_and_lock(self):
        files = {'package.json':json.dumps({'name':'p','devDependencies':{'risk':'^0.13.15'}})}
        p = m.parse_projects(snapshot('a',files),'npm','p')[0]
        obs = m.observation(p,'risk','0','6.3.14')
        self.assertTrue(obs['affected']); self.assertEqual(obs['dependency_scope'],'development')
        files['package-lock.json'] = json.dumps({'lockfileVersion':3,'packages':{'':{'devDependencies':{'risk':'^0.13.15'}},'node_modules/risk':{'version':'0.13.16'}}})
        p = m.parse_projects(snapshot('b',files),'npm','p')[0]
        self.assertEqual(m.observation(p,'risk','0','6.3.14')['version'],'0.13.16')

    def test_python_extras_and_companion_do_not_activate_environment(self):
        files = {'setup.cfg':'[metadata]\nname=p\n[options]\ninstall_requires=other==1.0\n[options.extras_require]\ndeploy= risk~=0.19\ndev= risk==0.19.2\n', 'requirements.txt':'risk==0.19.2\n'}
        p = m.parse_projects(snapshot('a',files),'PyPI','p')[0]
        obs = m.observation(p,'risk','0','1.2.2')
        self.assertTrue(obs['affected']); self.assertIsNone(obs['version'])
        self.assertEqual({r['scope'] for r in obs['declarations']}, {'extras','default'})

    def test_cross_scope_different_versions_are_not_intersected(self):
        files = {'pyproject.toml':'[project]\nname="p"\ndependencies=["risk==1.0"]\n[project.optional-dependencies]\nextra=["risk==2.0"]\n'}
        p = m.parse_projects(snapshot('a',files),'PyPI','p')[0]
        self.assertIsNone(m.observation(p,'risk','0','1.5')['affected'])

    def test_python_static_setup_extras(self):
        files = {'setup.py':'from setuptools import setup\nsetup(name="p", extras_require={"dev":["risk==1.0"]})\n'}
        p = m.parse_projects(snapshot('a',files),'PyPI','p')[0]
        self.assertTrue(m.observation(p,'risk','0','1.5')['affected'])

    def test_maven_parent_and_child_remain_separate(self):
        parent = '<project><groupId>g</groupId><artifactId>p</artifactId><version>1</version><packaging>pom</packaging><dependencyManagement><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId><version>1.0</version></dependency></dependencies></dependencyManagement></project>'
        child = '<project><parent><groupId>g</groupId><artifactId>p</artifactId><version>1</version></parent><artifactId>child</artifactId><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId><scope>test</scope></dependency></dependencies></project>'
        s = snapshot('a', {'pom.xml':parent,'child/pom.xml':child})
        p = m.parse_projects(s,'Maven','g:p')[0]
        c = m.parse_projects(s,'Maven','g:child')[0]
        self.assertIsNone(m.observation(p,'x:risk','0','1.5')['affected'])
        self.assertTrue(m.observation(c,'x:risk','0','1.5')['affected'])
        self.assertEqual(m.observation(c,'x:risk','0','1.5')['dependency_scope'],'test')
        self.assertEqual(m.parse_projects(s,'Maven','g:historical'), [])

    def test_affected_range_can_have_known_major_without_invented_pin(self):
        a = m.parse_projects(snapshot('a', {'package.json':'{"name":"p","dependencies":{"risk":"^1.0.0"}}'}),'npm','p')[0]
        b = m.parse_projects(snapshot('b', {'package.json':'{"name":"p","dependencies":{"risk":"2.0.0"}}'}),'npm','p')[0]
        self.assertEqual(m.transition(a,b,'risk','0','2.0.0')['state'],2)

    def test_typed_form_rejects_open_enum_boolean_and_seventh_key(self):
        form = {'case_results':[dict(dependency_scope=None,version_evidence=empty_version_evidence(),endpoint_state=None,first_response_state=None)]}
        self.assertTrue(validate_form(form))
        for field, value in [('dependency_scope','human_to_record'),('endpoint_state',False),('version_evidence',{})]:
            bad=copy.deepcopy(form);bad['case_results'][0][field]=value
            with self.assertRaises(ValueError):validate_form(bad)
        self.assertEqual(len(SCOPES),12)


if __name__ == '__main__':
    unittest.main()
