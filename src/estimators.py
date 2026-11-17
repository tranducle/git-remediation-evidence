"""Reference estimators and synthetic populations with known effects."""
import numpy as np


def cluster_variance(influence, repository, cve, adjust=True):
    influence=np.asarray(influence,dtype=float)
    if not np.isfinite(influence).all():
        raise ValueError('Non-finite influence contribution')
    def component(labels):
        _, index=np.unique(labels,return_inverse=True)
        groups=index.max()+1
        if groups<2:
            raise ValueError('At least two independent clusters required')
        totals=np.bincount(index,weights=influence)
        return np.dot(totals,totals)*(groups/(groups-1) if adjust else 1)
    pairs=np.array([str(a)+':'+str(b) for a,b in zip(repository,cve)])
    variance=component(repository)+component(cve)-component(pairs)
                                                                              
    return float(variance)


def weighted_mean_influence(values, selected, weights):
    weights=np.where(selected,weights,0.)
    total=weights.sum()
    if total<=0 or np.count_nonzero(weights)<2:
        raise ValueError('Insufficient common-support observations')
    if not np.isfinite(values[weights>0]).all():
        raise ValueError('Unknown observed outcome cannot be coded zero')
    safe=np.where(weights>0,values,0.)
    mean=np.dot(weights,safe)/total
    influence=weights*(safe-mean)/total
    return float(mean),influence


def did_cell(y, cohort, group, event_time, *, anticipation=0, eligibility=None,
             observed=None, observation_probability=None, post_selection=False):
    t=group+event_time
    baseline=group-anticipation-1
    if min(t,baseline)<0 or max(t,baseline)>=y.shape[1]:
        raise ValueError('Requested horizon is outside the observed panel')
    n=len(cohort)
    eligible=np.ones(n,dtype=bool) if eligibility is None else eligibility[:,group-1]
    treated=(cohort==group)&eligible
                                                                            
                                                                          
    controls=(cohort>max(t,baseline)+anticipation)&(cohort!=group)&eligible
    selected=treated|controls
    if observed is not None:
        selected &= observed[:,t]&observed[:,baseline]
    if post_selection:
        selected &= y[:,t]==0
    weights=np.ones(n)
    if observation_probability is not None:
        probability=observation_probability[:,t]*observation_probability[:,baseline]
        if np.any(probability<=0) or np.any(probability>1):
            raise ValueError('Observation probabilities must be in (0, 1]')
        weights=1/probability
    delta=y[:,t]-y[:,baseline]
    a,ia=weighted_mean_influence(delta,treated&selected,weights)
    b,ib=weighted_mean_influence(delta,controls&selected,weights)
    return dict(estimate=a-b,influence=ia-ib,treated_count=int(treated.sum()),
                control_count=int(controls.sum()),group=group,time=t,baseline=baseline,used=selected)


def did_aggregate(y,cohort,event_time,**kwargs):
    groups=sorted(int(x) for x in np.unique(cohort) if np.isfinite(x))
    cells=[did_cell(y,cohort,g,event_time,**kwargs) for g in groups]
    counts=np.array([cell['treated_count'] for cell in cells],dtype=float)
    weights=counts/counts.sum()
    return dict(estimate=float(sum(w*c['estimate'] for w,c in zip(weights,cells))),
                influence=sum(w*c['influence'] for w,c in zip(weights,cells)),
                weights=weights,cells=cells,used=np.any([c['used'] for c in cells],axis=0))


def aj_cif(time,status,horizon,entry=None,causes=3):
    pass
    time=np.asarray(time,dtype=float);status=np.asarray(status)
    entry=np.zeros(len(time)) if entry is None else np.asarray(entry,dtype=float)
    if not(len(time)==len(status)==len(entry)) or not len(time):
        raise ValueError('Cohort arrays have inconsistent or empty dimensions')
    if status.dtype.kind not in 'iu' or status.dtype.kind=='b' or np.any(status<0) or np.any(status>causes):
        raise ValueError('Event status must be integer 0 for censor or a supported cause')
    if not np.isfinite(time).all() or not np.isfinite(entry).all() or np.any(entry<0) or np.any(time<=entry):
        raise ValueError('Invalid time, entry, or unknown event evidence')
    survival=1.;cif=np.zeros(causes);ds=np.zeros(len(time));df=np.zeros((len(time),causes));trace=[]
    for t in np.unique(time[(status>0)&(time<=horizon)]):
        risk=(entry<t)&(time>=t);nr=risk.sum()
        if nr<=0:
            raise ValueError('Event has no valid risk set')
        events=np.column_stack([(time==t)&(status==cause) for cause in range(1,causes+1)])
        hazards=events.sum(axis=0)/nr
        derivative=(events-risk[:,None]*hazards)/nr
        df += ds[:,None]*hazards + survival*derivative
        ds=ds*(1-hazards.sum())-survival*derivative.sum(axis=1)
        cif += survival*hazards
        survival *= 1-hazards.sum()
        trace.append(dict(time=float(t),risk=int(nr),events=events.sum(axis=0).tolist(),
                          cif=cif.tolist(),survival=float(survival)))
    if np.any(cif < -1e-12) or survival < -1e-12 or abs(cif.sum()+survival-1)>1e-10:
        raise ArithmeticError('Probability accounting failed')
    return dict(cif=cif,influence=df,survival=float(survival),trace=trace)


def reversible_and_first(states):
    known={0,1,2,3,4}
    first=None;current=None
    for state in states:
        if state is None:
            current=None
            continue
        if type(state) is not int or state not in known:
            raise ValueError('Unknown states must remain explicit')
        current=state
        if first is None and state in {1,2,3,4}:
            first=state
    return dict(current_state=current,first_response_state=first)
