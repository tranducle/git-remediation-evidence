"""Test conservative parsing of immutable package metadata."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from history_measurement import parse_projects, observation, transition
from maven_models import PomStore
from version_rules import SemVer, ordered, affected, allows, enforces_fixed
from test_measurement import snapshot, npm


class EcosystemVersions(unittest.TestCase):
    def test_semver_spec_precedence_and_build_metadata(self):
        versions=['1.0.0-alpha','1.0.0-alpha.1','1.0.0-alpha.beta','1.0.0-beta','1.0.0-beta.2','1.0.0-beta.11','1.0.0-rc.1','1.0.0']
        for before,after in zip(versions,versions[1:]):
            self.assertLess(SemVer(before),SemVer(after))
        self.assertEqual(SemVer('1.0.0+build.1'),SemVer('1.0.0+build.2'))

    def test_semver_rejects_leading_zero_and_empty_identifier(self):
        for invalid in ['01.0.0','1.0.0-01','1.0.0-a..b','1.0.0+','1.0','1.0.0+bad..id']:
            self.assertIsNone(ordered(invalid,'npm'))

    def test_npm_prerelease_is_not_guessed_as_stable(self):
        self.assertTrue(affected('1.1.0-rc.1','1.0.0','1.1.0','npm'))
        self.assertFalse(allows('^1.0.0','1.1.0-rc.1','npm'))

    def test_pep440_epoch_dev_and_post(self):
        self.assertTrue(affected('1!1.0.dev1','0','1!1.0','PyPI'))
        self.assertFalse(affected('1.0.post1','0','1.0','PyPI'))
        self.assertTrue(allows('>=1.0,<2','1.0.post1','PyPI'))

    def test_maven_numeric_subset_and_qualifier_unknown(self):
        self.assertEqual(ordered('1.0','Maven'),ordered('1.0.0','Maven'))
        self.assertIsNone(ordered('1.0.Final','Maven'))
        self.assertIsNone(affected('1.0-SNAPSHOT','0','1.1','Maven'))

    def test_range_floor_and_disjunction(self):
        self.assertTrue(enforces_fixed('^1.2.0 || >=2.0.0 <3.0.0','1.1.0','npm'))
        self.assertFalse(enforces_fixed('^1.0.0 || ^2.0.0','1.1.0','npm'))
        self.assertIsNone(enforces_fixed('^1.2.0 garbage','1.1.0','npm'))
        self.assertTrue(allows('~1.2.0','1.2.8','npm'))
        self.assertFalse(allows('~1.2.0','1.3.0','npm'))


class PinnedMaven(unittest.TestCase):
    def test_unknown_profile_override_does_not_certify_the_default_pin(self):
        xml='<project><groupId>g</groupId><artifactId>consumer</artifactId><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId><version>1.0.0</version></dependency></dependencies><profiles><profile><id>other</id><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId><version>2.0.0</version></dependency></dependencies></profile></profiles></project>'
        p=parse_projects(snapshot('a',{'pom.xml':xml}),'Maven','g:consumer')[0]
        self.assertIsNone(observation(p,'x:risk','0','1.1.0')['affected'])

    def store(self, root, entries, available='2020-01-01T00:00:00Z'):
        index={}
        for n,(coordinate,content) in enumerate(entries.items()):
            path=Path(root)/(str(n)+'.pom');path.write_text(content)
            index[coordinate]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   available_from=available,source_url='https://repo.maven.apache.org/maven2/fixture')
        (Path(root)/'index.json').write_text(json.dumps(index))
        return PomStore(root)

    def test_imported_bom_and_explicit_local_override(self):
        bom='<project><groupId>b</groupId><artifactId>bom</artifactId><version>1</version><properties><risk.v>1.0.0</risk.v></properties><dependencyManagement><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId><version>${risk.v}</version></dependency></dependencies></dependencyManagement></project>'
        child='<project><groupId>g</groupId><artifactId>consumer</artifactId><dependencyManagement><dependencies><dependency><groupId>b</groupId><artifactId>bom</artifactId><version>1</version><type>pom</type><scope>import</scope></dependency><dependency><groupId>x</groupId><artifactId>risk</artifactId><version>1.0.1</version></dependency></dependencies></dependencyManagement><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId></dependency></dependencies></project>'
        with tempfile.TemporaryDirectory() as d:
            p=parse_projects(snapshot('a',{'pom.xml':child}),'Maven','g:consumer',pom_store=self.store(d,{'b:bom:1':bom}))[0]
            self.assertEqual(observation(p,'x:risk','0','1.1.0')['version'],'1.0.1')
            self.assertIn('@pom/b:bom:1',p['source_objects'])

    def test_external_parent_dependency_and_child_property(self):
        parent='<project><groupId>g</groupId><artifactId>parent</artifactId><version>1</version><properties><risk.v>1.0.0</risk.v></properties><dependencies><dependency><groupId>x</groupId><artifactId>risk</artifactId><version>${risk.v}</version></dependency></dependencies></project>'
        child='<project><parent><groupId>g</groupId><artifactId>parent</artifactId><version>1</version><relativePath/></parent><artifactId>consumer</artifactId><properties><risk.v>1.0.1</risk.v></properties></project>'
        with tempfile.TemporaryDirectory() as d:
            p=parse_projects(snapshot('a',{'pom.xml':child}),'Maven','g:consumer',pom_store=self.store(d,{'g:parent:1':parent}))[0]
            self.assertEqual(observation(p,'x:risk','0','1.1.0')['version'],'1.0.1')

    def test_cache_hash_and_historical_availability_fail_closed(self):
        pom='<project><groupId>g</groupId><artifactId>p</artifactId><version>1</version></project>'
        with tempfile.TemporaryDirectory() as d:
            store=self.store(d,{'g:p:1':pom},available='2030-01-01T00:00:00Z')
            with self.assertRaises(ValueError):store.get('g:p:1',1_700_000_000)
            (Path(d)/'0.pom').write_text(pom+' ')
            with self.assertRaises(ValueError):store.get('g:p:1',2_000_000_000)

    def test_missing_bom_does_not_establish_absence(self):
        text='<project><groupId>g</groupId><artifactId>consumer</artifactId><dependencyManagement><dependencies><dependency><groupId>b</groupId><artifactId>missing</artifactId><version>1</version><type>pom</type><scope>import</scope></dependency></dependencies></dependencyManagement></project>'
        p=parse_projects(snapshot('a',{'pom.xml':text}),'Maven','g:consumer')[0]
        self.assertIsNone(observation(p,'x:risk','0','1.1.0')['present'])

    def test_external_entities_are_rejected_and_diagnosed(self):
        diagnostics=[]
        p=parse_projects(snapshot('a',{'pom.xml':'<!DOCTYPE project [<!ENTITY e SYSTEM "file:/private">]><project/>'}),'Maven',diagnostics=diagnostics)
        self.assertEqual(p,[])
        self.assertTrue(diagnostics)


class FailClosedSources(unittest.TestCase):
    def test_duplicate_pnpm_keys_are_not_last_wins(self):
        diagnostics=[]
        files={'package.json':'{"name":"consumer","dependencies":{"risk":"^1.0.0"}}',
               'pnpm-lock.yaml':'importers:\n  .:\n    dependencies:\n      risk:\n        specifier: ^1.0.0\n        version: 1.0.0\n        version: 1.2.0\n'}
        self.assertEqual(parse_projects(snapshot('a',files),'npm',diagnostics=diagnostics),[])
        self.assertTrue(diagnostics)

    def test_git_or_local_npm_source_is_not_assumed_registry_exposure(self):
        p=npm('a','git+https://example.invalid/repo#tag','1.0.0')
        self.assertIsNone(observation(p,'risk','0','1.1.0')['affected'])

    def test_duplicate_manifest_keys_are_not_last_wins(self):
        diagnostics=[]
        p=parse_projects(snapshot('a',{'package.json':'{"name":"consumer","dependencies":{"risk":"1.0.0"},"dependencies":{}}'}),'npm',diagnostics=diagnostics)
        self.assertEqual(p,[])
        self.assertIn('Duplicate',diagnostics[0]['error'])

    def test_conditional_setup_cannot_establish_a_present_dependency(self):
        text='from setuptools import setup\nif condition:\n setup(name="consumer",install_requires=["risk==1.0.0"])\n'
        p=parse_projects(snapshot('a',{'setup.py':text}),'PyPI','consumer')[0]
        self.assertIsNone(observation(p,'risk','0','1.1.0')['present'])

    def test_unreviewed_removal_is_a_candidate_not_final_state_four(self):
        p=transition(npm('a'),npm('b',None),'risk','0','1.1.0')
        self.assertIsNone(p['state'])
        self.assertEqual(p['candidate_state'],4)

    def test_removal_audit_is_bound_to_the_commit_pair(self):
        audit=dict(old='risk',verified_no_successor=True,evidence_id='fixture-audit',before_commit='a',after_commit='b')
        self.assertEqual(transition(npm('a'),npm('b',None),'risk','0','1.1.0',[audit])['state'],4)
        audit['after_commit']='different'
        self.assertIsNone(transition(npm('a'),npm('b',None),'risk','0','1.1.0',[audit])['state'])

    def test_verified_alias_must_be_bound_and_safe(self):
        after=npm('b',None,extra={'successor':'1.1.0'})
        alias=dict(old='risk',new='successor',verified=True,evidence_id='fixture',comparable=True,
                   successor_fixed='1.1.0',before_commit='a',after_commit='b')
        self.assertEqual(transition(npm('a'),after,'risk','0','1.1.0',[alias])['state'],1)
        alias['before_commit']='different'
        self.assertIsNone(transition(npm('a'),after,'risk','0','1.1.0',[alias])['state'])
