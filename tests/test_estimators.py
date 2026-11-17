"""Test estimator invariants against exact and synthetic reference cases."""
from fractions import Fraction
import unittest
import numpy as np
from estimators import aj_cif, did_cell, did_aggregate, cluster_variance, reversible_and_first
from synthetic_cohorts import panel

COHORTS = {
 'censoring':dict(entry=[0,0,0,0],time=[1,2,3,4],status=[1,2,0,1],truth=[Fraction(3,4),Fraction(1,4),0]),
 'event_censor_ties':dict(entry=[0,0,0,0],time=[1,1,1,2],status=[1,2,0,1],truth=[Fraction(3,4),Fraction(1,4),0]),
 'delayed_entry':dict(entry=[0,0,1,2],time=[1,3,2,4],status=[1,2,0,1],truth=[Fraction(3,4),Fraction(1,4),0]),
 'all_censored':dict(entry=[0,0,0],time=[1,2,3],status=[0,0,0],truth=[0,0,0]),
 'removal_competes':dict(entry=[0,0,0,0],time=[1,2,3,4],status=[1,3,0,1],truth=[Fraction(3,4),0,Fraction(1,4)])}


class ProductIntegralTests(unittest.TestCase):
    def test_exact_fraction_cohorts(self):
        for name,case in COHORTS.items():
            result=aj_cif(case['time'],case['status'],180,entry=case['entry'])
            np.testing.assert_allclose(result['cif'],[float(x) for x in case['truth']],atol=1e-14,err_msg=name)
            np.testing.assert_allclose(result['influence'].sum(axis=0),0,atol=1e-14)

    def test_delayed_entry_is_not_counted_early(self):
        result=aj_cif([1,3,2,4],[1,2,0,1],180,entry=[0,0,1,2])
        self.assertEqual(result['trace'][0]['risk'],2)
        self.assertEqual(result['trace'][0]['cif'][0],.5)

    def test_competing_event_is_not_censoring(self):
        cif=aj_cif([1,2,3,4],[1,3,0,1],180)['cif'][2]
        naive=1/3
        self.assertEqual(cif,.25);self.assertNotEqual(cif,naive)

    def test_invalid_times_states_and_unknowns_fail(self):
        for times,status,entry in [([0],[1],[0]),([1],[True],[0]),([np.nan],[1],[0]),([1],[4],[0]),([1],[1],[2])]:
            with self.assertRaises(ValueError):aj_cif(times,status,180,entry=entry)

    def test_reversion_does_not_erase_first_response_or_unknown_gap(self):
        self.assertEqual(reversible_and_first([0,1,0]),dict(current_state=0,first_response_state=1))
        self.assertEqual(reversible_and_first([0,3,None]),dict(current_state=None,first_response_state=3))


class DIDTests(unittest.TestCase):
    def setUp(self):
        self.data=panel(42,'staggered_heterogeneous',nrepo=8,ncve=24)

    def test_already_treated_outcomes_cannot_change_later_cell(self):
        d=self.data;y=d['y'].copy()
        a=did_cell(y,d['cohort'],8,13)['estimate']
        y[d['cohort']==4,21]+=1000
        self.assertEqual(did_cell(y,d['cohort'],8,13)['estimate'],a)

    def test_lead_controls_exclude_target_cohort(self):
        d=self.data;r=did_cell(d['y'],d['cohort'],4,-3)
        self.assertEqual(r['control_count'],int((d['cohort']>4).sum()))

    def test_shared_weight_linearity_and_cohort_aggregation(self):
        d=self.data;a=did_aggregate(d['y'],d['cohort'],13)
        b=did_aggregate(1-d['y'],d['cohort'],13)
        c=did_aggregate(np.ones_like(d['y']),d['cohort'],13)
        self.assertAlmostEqual(a['weights'].sum(),1)
        self.assertAlmostEqual(a['estimate']+b['estimate'],c['estimate'])
        np.testing.assert_allclose(a['influence']+b['influence'],c['influence'],atol=1e-14)

    def test_missing_is_not_forward_filled_or_zero(self):
        d=self.data;y=d['y'].copy();y[0,17]=np.nan
                                                                               
        index=np.flatnonzero(d['cohort']==4)[0];y[index,17]=np.nan
        with self.assertRaises(ValueError):did_cell(y,d['cohort'],4,13)

    def test_cluster_inference_requires_clusters(self):
        with self.assertRaises(ValueError):cluster_variance([.1,-.1],[0,0],[0,1])

    def test_dgp_probabilities_are_known_and_reentry_occurs(self):
        data=panel(43,'reentry_current_state',nrepo=8,ncve=24)
        self.assertGreater(data['reentries'],0)
        self.assertTrue(np.isin(data['y'],[0,1]).all())


if __name__=='__main__':
    unittest.main()
