"""Label-free deployment artifact and fixed training-independent mixture draw."""
import hashlib
import math
import numpy as np
from h13_signal_features import vector

ACTIONS=['upgrade8','original_query','cached20']
ALL_ACTIONS=['stop4']+ACTIONS

def choose(observation,artifact):
    rule=artifact['rule']
    if rule['type']=='constant':return dict(action=ACTIONS[rule['action']],predicted_gains=None,head_flops=0)
    h,num=vector(observation,'draft_gain')
    scaled=(np.asarray(num)-artifact['mean'])/artifact['scale']/math.sqrt(len(num))
    x=np.r_[h,scaled];gain=np.asarray(artifact['coef'])@x+artifact['intercept']
    utility=np.r_[0,gain-rule['lambda']*np.maximum(artifact['extra_work_TF'],0)]
    action=int(utility.argmax())
    return dict(action=ACTIONS[action],predicted_gains=gain.tolist(),utilities=utility.tolist(),head_flops=artifact['head_flops'])

def mixture_action(dataset,qid,weights):
    assert len(weights)==4 and abs(sum(weights)-1)<1e-10 and min(weights)>=0
    digest=hashlib.sha256(('1311|'+dataset+'|'+qid).encode()).digest()
    u=int.from_bytes(digest[:8],'big')/2**64
    action=min(int(np.searchsorted(np.cumsum(weights),u,side='right')),3)
    return ALL_ACTIONS[action]

def constant_mix(f1,em,cost,budget):
    f1,em,cost=map(np.asarray,[f1,em,cost]);n=len(f1);candidates=[]
    for j in range(n):
        w=np.eye(n)[j]
        if cost[j]<=budget+1e-10:candidates.append(w)
        for k in range(j+1,n):
            gap=cost[k]-cost[j]
            if abs(gap)<1e-12:continue
            p=(budget-cost[j])/gap
            if 0<=p<=1:
                w=np.zeros(n);w[j]=1-p;w[k]=p;candidates.append(w)
    if not candidates:return dict(status='infeasible_budget')
    w=min(candidates,key=lambda w:(-float(w@f1),float(w@cost),tuple(-w)))
    return dict(status='optimal',weights=w.tolist(),f1=float(w@f1),em=float(w@em),work_TF=float(w@cost))
