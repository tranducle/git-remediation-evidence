"""Reference estimators and synthetic populations with known effects."""
import numpy as np

SCENARIOS = ['null','staggered_heterogeneous','anticipation_corrected','changing_risk_sets',
             'crossed_cve_repo_clusters','censoring_ipcw','reentry_current_state','competing_removal_cif']


def cluster_grid(nrepo,ncve):
    return np.repeat(np.arange(nrepo),ncve),np.tile(np.arange(ncve),nrepo)


def panel(seed,scenario,nrepo=32,ncve=120):
    rng=np.random.default_rng(seed);repo,cve=cluster_grid(nrepo,ncve)
    n=len(repo);nt=27
    allocation=np.repeat(np.array([4.,8.,12.,np.inf]),ncve//4)
    rng.shuffle(allocation);cohort=allocation[cve]
    x=rng.integers(0,2,size=n)
    amplitude=.12 if scenario=='crossed_cve_repo_clusters' else .04
    cve_shock=rng.uniform(-amplitude,amplitude,size=(ncve,nt))
    repo_shock=rng.uniform(-.06,.06,size=(nrepo,nt))
    base=.30+.04*x[:,None]+.001*np.arange(nt)[None,:]
    untreated=base+cve_shock[cve,:]+repo_shock[repo,:]
    effects=np.zeros((n,nt));group_effects={4:.025,8:.05,12:.075}
    is_null=scenario in ['null','crossed_cve_repo_clusters']
    for group,tau in group_effects.items():
        if is_null:
            tau=0.
        selected=cohort==group
        for t in range(nt):
            lag=t-group
            value=tau*min(1.,(lag+1)/7) if lag>=0 else .025 if scenario=='anticipation_corrected' and lag in [-2,-1] else 0.
            effects[selected,t]=value
    probability=untreated+effects
    if np.any(probability<=0) or np.any(probability>=1):
        raise ArithmeticError('Known DGP probability leaves (0,1); no clipping allowed')
    reentries=0
    if scenario=='reentry_current_state':
        persistence=.65
        y=np.zeros((n,nt));y[:,0]=rng.binomial(1,probability[:,0])
        marginal_effect=effects.copy()
        for t in range(1,nt):
            y[:,t]=rng.binomial(1,persistence*y[:,t-1]+(1-persistence)*probability[:,t])
            marginal_effect[:,t]=persistence*marginal_effect[:,t-1]+(1-persistence)*effects[:,t]
        effects=marginal_effect
        reentries=int(((y[:,1:]==0)&(y[:,:-1]==1)).sum())
    else:
        y=rng.binomial(1,probability).astype(float)
    eligibility=np.ones((n,nt),dtype=bool)
    if scenario=='changing_risk_sets':
        p=.60+.15*x[:,None]+.06*np.sin(np.arange(nt)[None,:]/3)
        eligibility=rng.random((n,nt))<p
    observed=np.ones((n,nt),dtype=bool);observation_probability=np.ones((n,nt))
    if scenario=='censoring_ipcw':
        observation_probability=np.broadcast_to(.55+.30*x[:,None],(n,nt)).copy()
        observed=rng.random((n,nt))<observation_probability
    return dict(y=y,cohort=cohort,repository=repo,cve=cve,effects=effects,eligibility=eligibility,
                observed=observed,observation_probability=observation_probability,reentries=reentries)


def true_att(data,event_time,anticipation=0):
    values=[];counts=[]
    for group in [4,8,12]:
        selected=(data['cohort']==group)&data['eligibility'][:,group-1]
        values.append(float(np.mean(data['effects'][selected,group+event_time]-data['effects'][selected,group-anticipation-1])))
        counts.append(selected.sum())
    return float(np.average(values,weights=counts))


def competing(seed,nrepo=32,ncve=120):
    rng=np.random.default_rng(seed);repo,cve=cluster_grid(nrepo,ncve);n=len(repo)
    exposed=np.repeat([0,1],ncve//2);rng.shuffle(exposed);treated=exposed[cve]
    zc=rng.choice([-1,1],size=ncve);zr=rng.choice([-1,1],size=nrepo)
    frailty=.005*zc[cve]+.003*zr[repo]
    hazards=np.column_stack([.08+frailty,.04+frailty/2,.02+.02*treated+frailty])
    time=np.full(n,180.,dtype=float);status=np.zeros(n,dtype=int);alive=np.ones(n,dtype=bool)
    for day in [30,60,90,120,150,180]:
        draw=rng.random(n);cutoffs=hazards.cumsum(axis=1)
        for cause in [1,2,3]:
            hit=alive&(draw<cutoffs[:,cause-1])&(draw >= (cutoffs[:,cause-2] if cause>1 else 0))
            time[hit]=day;status[hit]=cause
        alive &= draw>=cutoffs[:,-1]
    censor=rng.exponential(700,size=n)
    lost=censor<time;time[lost]=censor[lost];status[lost]=0
    truth=[]
    for treatment in [0,1]:
        values=[]
        for a in [-1,1]:
            for b in [-1,1]:
                f=.005*a+.003*b;h=np.array([.08+f,.04+f/2,.02+.02*treatment+f])
                values.append(h/h.sum()*(1-(1-h.sum())**6))
        truth.append(np.mean(values,axis=0))
    return dict(time=time,status=status,treated=treated,repository=repo,cve=cve,
                truth=truth[1]-truth[0],horizon=180)
