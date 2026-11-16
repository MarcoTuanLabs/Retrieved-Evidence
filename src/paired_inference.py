"""Paired source-cluster intervals and explicit quality/cost gates.

Repeated timings belong to a question; they never increase independent sample n.
The caller chooses and freezes policies before passing confirmation outcomes.
"""
import numpy as np

def interval(x):return [float(v) for v in np.quantile(x,[.025,.975])]

class PairedClusters:
    def __init__(self,groups,draws=10000,seed=1307):
        self.groups=np.asarray(groups);self.n=len(self.groups)
        unique=sorted(set(self.groups.tolist()));assert len(unique)>=2
        self.members=[np.flatnonzero(self.groups==g) for g in unique]
        self.sizes=np.asarray([len(v) for v in self.members])
        self.draws=np.random.default_rng(seed).integers(0,len(unique),(draws,len(unique)))
        self.denominator=self.sizes[self.draws].sum(1)
        self.seed=seed

    def bootstrap_means(self,x):
        x=np.asarray(x,dtype=float);assert x.shape==(self.n,) and np.isfinite(x).all()
        sums=np.asarray([x[m].sum() for m in self.members])
        return sums[self.draws].sum(1)/self.denominator

    def compare(self,candidate,reference,quality_margin=.02,work_target=.25,time_target=.10,p95_guard=.10):
        required={'f1','em','work_TF','seconds'}
        assert required<=set(candidate) and required<=set(reference)
        c={};r={}
        for field in required:
            a,b=np.asarray(candidate[field],dtype=float),np.asarray(reference[field],dtype=float)
            assert a.ndim==b.ndim==2 and a.shape==b.shape and a.shape[0]==self.n
            assert np.isfinite(a).all() and np.isfinite(b).all()
            if field in ('f1','em'):assert ((a>=0)&(a<=1)&(b>=0)&(b<=1)).all()
            else:assert (a>0).all() and (b>0).all()
            c[field],r[field]=a.mean(1),b.mean(1)
        result=dict(questions=self.n,source_clusters=len(self.members),bootstrap_draws=len(self.draws),seed=self.seed,
                    repeated_measurements_counted_as_independent=False,comparisons={})
        for field in ['f1','em']:
            d=c[field]-r[field];ci=np.asarray(interval(self.bootstrap_means(d)))
            result['comparisons'][field]=dict(candidate_mean=float(c[field].mean()),reference_mean=float(r[field].mean()),
                delta_pp=float(d.mean()*100),paired95_ci_pp=(ci*100).tolist(),
                noninferior=bool(ci[0]>=-quality_margin),superior=bool(ci[0]>0),margin_pp=quality_margin*100)
        for field in ['work_TF','seconds']:
            ratio=float(c[field].mean()/r[field].mean())
            draws=self.bootstrap_means(c[field])/self.bootstrap_means(r[field])
            result['comparisons'][field]=dict(candidate_mean=float(c[field].mean()),reference_mean=float(r[field].mean()),
                ratio=ratio,reduction_percent=(1-ratio)*100,paired95_ratio_ci=interval(draws))
        p95c=float(np.quantile(np.asarray(candidate['seconds']),.95));p95r=float(np.quantile(np.asarray(reference['seconds']),.95))
        result['p95_latency']=dict(candidate_seconds=p95c,reference_seconds=p95r,ratio=p95c/p95r,
                                   scope='Raw request repetitions; guard uses point ratio, not an independent-sample interval')
        gates=dict(f1_noninferiority=result['comparisons']['f1']['noninferior'],em_noninferiority=result['comparisons']['em']['noninferior'],
            mean_work=result['comparisons']['work_TF']['ratio']<=1-work_target,
            mean_latency=result['comparisons']['seconds']['ratio']<=1-time_target,
            p95_latency=p95c/p95r<=1+p95_guard)
        result['gates']=gates;result['all_gates_pass']=all(gates.values())
        result['cost_gate_scope']='Point reductions; paired uncertainty reported separately. Quality gates use95%source-cluster lower bounds.'
        return result
