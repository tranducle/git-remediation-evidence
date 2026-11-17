"""Run the known-process Monte Carlo calibration and retain every replicate."""
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from scipy.stats import t as student_t
from estimators import did_aggregate, aj_cif, cluster_variance
from synthetic_cohorts import SCENARIOS, panel, true_att, competing

HERE=Path(__file__).resolve().parent


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def interval(estimate,variance,truth,critical):
    if not np.isfinite(variance) or variance<=0:
        return dict(variance=float(variance) if np.isfinite(variance) else None,
                    standard_error=None,ci_lower=None,ci_upper=None,coverage=False,reject_zero=False,valid=False)
    se=float(np.sqrt(variance));lo=float(estimate-critical*se);hi=float(estimate+critical*se)
    return dict(variance=float(variance),standard_error=se,ci_lower=lo,ci_upper=hi,
                coverage=bool(lo<=truth<=hi),reject_zero=bool(not(lo<=0<=hi)),valid=True)


def panel_replicate(seed,scenario,config):
    data=panel(seed,scenario,config['repository_clusters'],config['cve_clusters'])
    anticipation=2 if scenario=='anticipation_corrected' else 0
    kwargs=dict(anticipation=anticipation,eligibility=data['eligibility'],observed=data['observed'],
                observation_probability=data['observation_probability'] if scenario=='censoring_ipcw' else None)
    events=[];main=None
    critical=float(student_t.ppf(.975,min(config['repository_clusters'],config['cve_clusters'])-1))
    for lag in config['event_times']:
        result=did_aggregate(data['y'],data['cohort'],lag,**kwargs)
        truth=true_att(data,lag,anticipation);used=result['used'];phi=result['influence']
        variance=cluster_variance(phi[used],data['repository'][used],data['cve'][used])
        event=dict(event_time=lag,estimate=result['estimate'],truth=truth,bias=result['estimate']-truth,
                   **interval(result['estimate'],variance,truth,critical))
        if all(cell['time']==cell['baseline'] for cell in result['cells']):
            event.update(valid=True,variance=0.,standard_error=0.,ci_lower=0.,ci_upper=0.,
                         coverage=True,reject_zero=False,interval_type='normalized_reference_no_inference')
        events.append(event)
        if lag==13:
            main=event.copy();primary=result
    main.update(aux_estimate=None,aux_truth=None,aux_bias=None,aux_coverage=None,aux_method=None,reentries=data['reentries'])
    if scenario=='anticipation_corrected':
        args=dict(kwargs,anticipation=0)
        aux=did_aggregate(data['y'],data['cohort'],13,**args)['estimate'];name='unadjusted_anticipatory_baseline'
    elif scenario=='changing_risk_sets':
        aux=did_aggregate(data['y'],data['cohort'],13,**kwargs,post_selection=True)['estimate'];name='post_outcome_risk_selection'
    elif scenario=='censoring_ipcw':
        rng=np.random.default_rng(seed+9000000)
        observed=rng.random(data['y'].shape)<np.where(data['y']==1,.25,.90)
        args=dict(kwargs,observed=observed,observation_probability=None)
        aux=did_aggregate(data['y'],data['cohort'],13,**args)['estimate'];name='outcome_dependent_censoring_without_adjustment'
    elif scenario=='reentry_current_state':
        absorbed=np.maximum.accumulate(data['y'],axis=1)
        aux=did_aggregate(absorbed,data['cohort'],13,**kwargs)['estimate'];name='absorbing_measure_substituted_for_current_state'
    elif scenario=='crossed_cve_repo_clusters':
        used=primary['used'];phi=primary['influence'][used]
        variance=float(np.dot(phi,phi)*len(phi)/(len(phi)-1))
        aux=primary['estimate'];name='independent_row_interval_under_crossed_clusters'
        main['aux_coverage']=interval(aux,variance,main['truth'],1.959963984540054)['coverage']
    else:
        return main,events
    main.update(aux_estimate=aux,aux_truth=main['truth'],aux_bias=aux-main['truth'],aux_method=name)
    return main,events


def cif_replicate(seed,config):
    data=competing(seed,config['repository_clusters'],config['cve_clusters'])
    fits=[]
    for group in [0,1]:
        selected=data['treated']==group
        fits.append(aj_cif(data['time'][selected],data['status'][selected],180))
    estimate=fits[1]['cif']-fits[0]['cif'];phi=np.zeros((len(data['time']),3))
    for group in [0,1]:
        phi[data['treated']==group]=(1 if group else -1)*fits[group]['influence']
    critical=float(student_t.ppf(.975,min(config['repository_clusters'],config['cve_clusters'])-1))
    events=[]
    for cause in range(3):
        variance=cluster_variance(phi[:,cause],data['repository'],data['cve'])
        events.append(dict(event_time=180,cause=cause+1,estimate=float(estimate[cause]),truth=float(data['truth'][cause]),
             bias=float(estimate[cause]-data['truth'][cause]),**interval(estimate[cause],variance,data['truth'][cause],critical)))
    main=events[2].copy()
    main.update(aux_estimate=None,aux_truth=None,aux_bias=None,aux_coverage=None,aux_method=None,reentries=None)
    return main,events


def wilson(count,n):
    z=1.959963984540054;p=count/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]


def summarize(rows,scenario,config):
    n=len(rows);coverage=sum(row['coverage'] for row in rows);rejections=sum(row['reject_zero'] for row in rows)
    bias=np.array([row['bias'] for row in rows]);mme=.035 if scenario=='competing_removal_cif' else .05
    valid=sum(row['valid'] for row in rows)
    result=dict(scenario=scenario,requested_replicates=config['replicates'],retained_replicates=n,valid_replicates=valid,
                coverage=coverage/n,coverage_MCSE=float(np.sqrt(coverage/n*(1-coverage/n)/n)),coverage_Wilson95=wilson(coverage,n),
                mean_bias=float(bias.mean()),bias_MCSE=float(bias.std(ddof=1)/np.sqrt(n)) if n>1 else None,
                mean_estimate=float(np.mean([row['estimate'] for row in rows])),mean_truth=float(np.mean([row['truth'] for row in rows])),
                rejection_rate=rejections/n,rejection_Wilson95=wilson(rejections,n),MME=mme,bias_limit=.1*mme)
    checks=dict(all_requested_retained=n==config['replicates'],all_variances_valid=valid==n,
                coverage_in_frozen_band=.92<=coverage/n<=.98,bias_below_ten_percent_MME=abs(result['mean_bias'])<.1*mme)
    if scenario in ['null','crossed_cve_repo_clusters']:
        checks['null_rejection_at_most_008']=rejections/n<=.08
    aux=[row for row in rows if row['aux_method']]
    if aux:
        result['stress_method']=aux[0]['aux_method']
        if scenario=='crossed_cve_repo_clusters':
            result['stress_coverage']=sum(row['aux_coverage'] for row in aux)/n
            result['stress_failure_exposed']=result['stress_coverage']<.92
        else:
            result['stress_mean_bias']=float(np.mean([row['aux_bias'] for row in aux]))
            result['stress_failure_exposed']=abs(result['stress_mean_bias'])>.1*mme
        checks['deliberate_violation_exposes_failure']=result['stress_failure_exposed']
    result['checks']=checks
    result['validation_status']='NOT_ASSESSED_SANITY' if config['mode']=='sanity' else 'PASS' if all(checks.values()) else 'FAIL_REPAIR'
    return result


def main():
    config_path=Path(sys.argv[1]).resolve();config=json.loads(config_path.read_text());output=Path(sys.argv[2]).resolve()
    output.mkdir(exist_ok=False)
    if config['mode']=='validation' and config['replicates']!=1000:
        raise ValueError('Frozen validation requires 1000 replicates per scenario')
    if config['scenarios']!=SCENARIOS:
        raise ValueError('Scenario list differs from frozen suite')
    started=time.monotonic();all_rows=[];event_rows=[];summaries=[]
    for scenario_index,scenario in enumerate(config['scenarios']):
        rows=[]
        for replicate in range(config['replicates']):
            seed=config['base_seed']+scenario_index*100000+replicate
            main_row,events=cif_replicate(seed,config) if scenario=='competing_removal_cif' else panel_replicate(seed,scenario,config)
            metadata=dict(scenario=scenario,replicate=replicate,seed=seed)
            rows.append(dict(**metadata,**main_row))
            event_rows += [dict(**metadata,**event) for event in events]
            if (replicate+1)%50==0:
                progress=dict(scenario=scenario,completed_in_scenario=replicate+1,retained_so_far=len(all_rows)+len(rows),
                     total=config['replicates']*len(SCENARIOS),elapsed_seconds=time.monotonic()-started,heartbeat=dt.datetime.now(dt.timezone.utc).isoformat())
                save(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        summary=summarize(rows,scenario,config);summaries.append(summary);all_rows+=rows
        save(output/(scenario+'_summary.json'),summary)
    for name,rows in [('replicates.csv',all_rows),('event_time_and_cause_diagnostics.csv',event_rows)]:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    receipt=dict(process_status='COMPLETED',validation_status='NOT_ASSESSED_SANITY' if config['mode']=='sanity' else 'PASS' if all(s['validation_status']=='PASS' for s in summaries) else 'FAIL_REPAIR',
         requested_replicates=config['replicates']*len(SCENARIOS),retained_replicates=len(all_rows),scenario_summaries=summaries,
         config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-started,
         source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
         environment={name:importlib.metadata.version(name) for name in ['numpy','scipy','statsmodels','pandas']},
         validation_scope='Unadjusted group-time DID, fixed-weight event aggregation, known observation weights, two-way influence uncertainty, reversible-state and first-response primitives, and unadjusted competing-risk CIF; no empirical identification or fitted DR nuisance-model validation')
    save(output/'summary.json',receipt)
    print(json.dumps({key:receipt[key] for key in ['process_status','validation_status','retained_replicates','elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
