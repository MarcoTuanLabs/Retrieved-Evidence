"""Frozen replay and official scoring of independent online confirmation requests."""
import hashlib
import importlib.util
import json
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from paired_inference import PairedClusters,interval

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/h31_fixed_policy_confirmation'


def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def scientific(value):
    if isinstance(value,list):return [scientific(v) for v in value]
    if isinstance(value,dict):return {k:scientific(v) for k,v in value.items() if k not in ['_messages','prefix_seconds','prefix_cpu_seconds','ranking_seconds','ranking_cpu_seconds','full_seconds','full_cpu_seconds']}
    return value


def main():
    start=time.perf_counter();lock=json.loads((RUN/'FROZEN.json').read_text());cfg=lock['configuration']
    for path,h in lock['hashes'].items():assert sha(ROOT/path)==h,path
    assert json.loads((RUN/'RESOURCE_STOP.json').read_text())['provider_state']=='stopped'
    collection=json.loads((RUN/'COLLECTION.json').read_text());assert collection['verified'] and collection['vault_archive_verified']
    cap=RUN/'downloads/capture';payload=RUN/'payload';receipt=json.loads((cap/'receipt.json').read_text())
    assert receipt['state']=='complete' and receipt['completed']==3200 and receipt['completed_questions']==800
    for path,h in json.loads((cap/'BACKUP.json').read_text())['files'].items():assert sha(cap/path)==h,path
    sys.path[:0]=[str(payload/'native/src'),str(payload),str(ROOT/'src')]
    from transformers import AutoTokenizer
    from costs_v2 import DenseForwardCost
    from h25_replay import Verifier
    from h31_pipeline import METHODS,execute
    assert METHODS==cfg['methods']
    tokenizers={role:AutoTokenizer.from_pretrained(payload/folder,local_files_only=True) for role,folder in [('qwen4b','model4_metadata'),('qwen8b','model8_metadata')]}
    costs={}
    for role in tokenizers:
        meta=json.loads((cap/'runtime'/(role+'.json')).read_text());spec=cfg['models'][role]
        assert meta['model']==spec['model'] and meta['revision']==spec['revision'] and meta['model_parameters']==spec['total_parameters']
        costs[role]=DenseForwardCost.from_config_total(meta['model_config'],meta['model_parameters'])
    verifier=Verifier(cap/'requests',tokenizers,costs,AutoTokenizer.from_pretrained(payload/'reranker',local_files_only=True),json.loads((payload/'reranker/config.json').read_text()))
    observations={(r['dataset'],r['id']):r for r in read(payload/'observations.jsonl')}
    records=read(cap/'records.jsonl');by={(r['dataset'],r['id'],r['method']):r for r in records}
    assert len(observations)==800 and len(records)==len(by)==3200
    assert set(by)=={(ds,q,m) for ds,q in observations for m in METHODS}
    rng=random.Random(cfg['seed']);keys=sorted(observations);rng.shuffle(keys);schedule=[]
    for ds,q in keys:
        methods=list(METHODS);rng.shuffle(methods);schedule.append(dict(dataset=ds,id=q,methods=methods))
    assert schedule==json.loads((cap/'schedule.json').read_text())
    assert [(r['dataset'],r['id'],r['method']) for r in records]==[(e['dataset'],e['id'],m) for e in schedule for m in e['methods']]
    databases={ds:sqlite3.connect('file:'+str(payload/spec['index'])+'?mode=ro',uri=True) for ds,spec in cfg['datasets'].items()}
    total_model_work=0.;writers=readers=rules=rankings=0
    for record in records:
        assert time.perf_counter()-start<1800
        ds,q,m=record['dataset'],record['id'],record['method'];queries=iter(record['physical_acquisitions'])
        def search(query,k,excluded):return verifier.retrieval(ds,databases[ds],query,k,excluded,next(queries),cfg['datasets'][ds]['index_sha256'])
        small=SimpleNamespace(tokenizer=tokenizers['qwen4b'],answer=lambda messages:SimpleNamespace(value=verifier.call('qwen4b',messages,record['writer'])),
            rank=lambda value:SimpleNamespace(value=verifier.rank(value,record['legacy_prefix']['ranking'])))
        big=SimpleNamespace(tokenizer=tokenizers['qwen8b'],answer=lambda messages:SimpleNamespace(value=verifier.call('qwen8b',messages,record['answer'])))
        recreated=execute(observations[ds,q],m,search,small,big)
        assert next(queries,None) is None and scientific(recreated)==scientific(record),(ds,q,m)
        assert record['full_seconds']>0 and record['full_cpu_seconds']>=0
        assert record['full_seconds']>=sum(c['wall_seconds'] for c in record['physical_acquisitions'])
        if record['accepted_by_rule']:assert record['total_model_flops']==0 and record['writer'] is None
        writers+=record['writer'] is not None;readers+=not record['accepted_by_rule'];rules+=record['accepted_by_rule']
        rankings+=record['legacy_prefix'] is not None and record['legacy_prefix']['ranking'] is not None
        total_model_work+=record['total_model_flops']
    for db in databases.values():db.close()
    assert verifier.calls==writers+readers==receipt['completed_writers']+receipt['completed_readers']
    assert rules==receipt['completed_rule_answers'] and rankings==verifier.rankings==receipt['completed_rankings']
    assert verifier.retrievals==receipt['physical_sparse_queries'] and np.isclose(total_model_work,receipt['campaign_main_model_flops'],rtol=1e-12,atol=0)
    replay=dict(status='passed',requests=3200,model_calls=verifier.calls,rankings=verifier.rankings,
        physical_queries_verified=verifier.retrievals,unique_SQL_reexecutions=len(verifier.search_cache),max_BM25_score_error=verifier.max_score_difference,
        full_policy_and_prompt_reconstruction=True,prefix_charged_once=True,no_gold_in_runtime=True,elapsed_seconds=time.perf_counter()-start)
    out=RUN/'results';out.mkdir(exist_ok=False);save(out/'replay_verification.json',replay)
    labels={(r['dataset'],r['id']):r for r in read(ROOT/'data/h31_confirmation/sealed/evaluator.jsonl')}
    assert set(labels)==set(observations)
    path=ROOT/'data/twowiki_source/official/2wikimultihop_evaluate_v1.1.py'
    spec=importlib.util.spec_from_file_location('h31_two_official',path);two=importlib.util.module_from_spec(spec);spec.loader.exec_module(two)
    scored=[]
    for r in records:
        label=labels[r['dataset'],r['id']];refs=[label['answer']]+label['answer_aliases'];answer=r['answer']['text']
        f1=max(two.f1_score(answer,g)[0] for g in refs);em=float(any(two.exact_match_score(answer,g) for g in refs))
        writer=r['writer'];ranker=r['legacy_prefix']['ranking'] if r['legacy_prefix'] else None
        scored.append(dict(dataset=r['dataset'],id=r['id'],method=r['method'],answer=answer,f1=float(f1),em=em,source_group=label['source_group'],type=label['type'],
            work_TF=r['total_model_flops']/1e12,seconds=r['full_seconds'],process_cpu_seconds=r['full_cpu_seconds'],
            reader_TF=0. if r['accepted_by_rule'] else r['answer']['model_flops']/1e12,
            writer_TF=writer['model_flops']/1e12 if writer else 0.,ranker_TF=ranker['estimated_linear_attention_flops']/1e12 if ranker else 0.,
            rule_accepted=r['accepted_by_rule'],rule_kind=r['rule']['parsed']['kind'] if r['rule'] and r['rule']['parsed'] else None,
            physical_queries=len(r['physical_acquisitions']),retrieval_seconds=sum(a['wall_seconds'] for a in r['physical_acquisitions']),
            any_context_cap=bool((r['visibility'] and (r['visibility']['truncated_document_ids'] or r['visibility']['omitted_document_ids'])) or (writer and (writer['visibility']['truncated_document_ids'] or writer['visibility']['omitted_document_ids']))),
            answer_cap=bool(not r['accepted_by_rule'] and r['answer']['generated_tokens']==128 and not r['answer']['ended_with_eos'])))
    sb={(r['id'],r['method']):r for r in scored};ids=sorted(q for ds,q in labels)
    cluster=PairedClusters([labels['twowiki',q]['source_group'] for q in ids],draws=10000,seed=1331)
    fields=['f1','em','work_TF','seconds','reader_TF','writer_TF','ranker_TF','physical_queries','retrieval_seconds','process_cpu_seconds']
    values={m:{f:np.array([sb[q,m][f] for q in ids]) for f in fields} for m in METHODS}
    table={m:{f:float(v.mean()) for f,v in fs.items()} for m,fs in values.items()}
    for m,t in table.items():
        rs=[sb[q,m] for q in ids];accepted=[r for r in rs if r['rule_accepted']]
        t.update(exact_count=int(sum(r['em'] for r in rs)),p95_seconds=float(np.quantile(values[m]['seconds'],.95)),
            accepted=len(accepted),correct_accepted=int(sum(r['em'] for r in accepted)),accepted_by_kind=dict(Counter(r['rule_kind'] for r in accepted)),
            context_caps=sum(r['any_context_cap'] for r in rs),answer_caps=sum(r['answer_cap'] for r in rs))
    comparisons={}
    for a,b in [('prefix_only','bm25_rewrite'),('relation_following','prefix_only'),('relation_following','bm25_rewrite'),('prefix_only','bm25_direct')]:
        c={}
        for f in ['f1','em']:
            delta=values[a][f]-values[b][f]
            c[f]=dict(delta_pp=float(100*delta.mean()),paired95_ci_pp=interval(100*cluster.bootstrap_means(delta)),gains=int((delta>1e-12).sum()),harms=int((delta< -1e-12).sum()))
        for f in ['work_TF','seconds']:
            c[f]=dict(ratio=float(values[a][f].mean()/values[b][f].mean()),paired95_ratio_ci=interval(cluster.bootstrap_means(values[a][f])/cluster.bootstrap_means(values[b][f])))
        c['p95_ratio']=table[a]['p95_seconds']/table[b]['p95_seconds'];comparisons[a+'_minus_'+b]=c
    primary=comparisons['prefix_only_minus_bm25_rewrite']
    joint=dict(f1_positive_lower_bound=primary['f1']['paired95_ci_pp'][0]>0,em_positive_lower_bound=primary['em']['paired95_ci_pp'][0]>0,
        model_work_upper_ratio_below_one=primary['work_TF']['paired95_ratio_ci'][1]<1,actual_latency_upper_ratio_below_one=primary['seconds']['paired95_ratio_ci'][1]<1,
        p95_point_ratio_at_most_1_1=primary['p95_ratio']<=1.1)
    margins={str(m):{f:primary[f]['paired95_ci_pp'][0]>=-m for f in ['f1','em']} for m in [.5,1.,2.,3.]}
    strata={typ:{m:dict(questions=sum(labels['twowiki',q]['type']==typ for q in ids),
        exact_count=int(sum(sb[q,m]['em'] for q in ids if labels['twowiki',q]['type']==typ)),
        accepted=sum(sb[q,m]['rule_accepted'] for q in ids if labels['twowiki',q]['type']==typ)) for m in METHODS} for typ in sorted({r['type'] for r in labels.values()})}
    selection=[]
    for a,b,limit in [('prefix_only','bm25_rewrite',3),('relation_following','prefix_only',2)]:
        for sign,name in [(1,'rescue'),(-1,'harm')]:
            qs=[q for q in ids if sign*(sb[q,a]['em']-sb[q,b]['em'])>0]
            qs.sort(key=lambda q:(not sb[q,a]['rule_accepted'],-abs(sb[q,a]['f1']-sb[q,b]['f1']),hashlib.sha256(q.encode()).hexdigest()))
            selection.extend(dict(id=q,dataset='twowiki',candidate=a,reference=b,outcome=name) for q in qs[:limit])
    (out/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    save(RUN/'CASE_SELECTION.json',selection)
    save(out/'summary.json',dict(questions=800,source_groups=800,methods=table,comparisons=comparisons,strata=strata,primary_joint_conditions=joint,
        primary_joint_improvement_confirmed=all(joint.values()),primary_margin_sensitivity=margins,
        official_scorer=dict(path=str(path),sha256=sha(path)),replay=replay,elapsed_seconds=time.perf_counter()-start,
        scope='Locked unseen-question/evaluated-support-title confirmation on balanced2Wiki only; actual independent request timing, single run per endpoint; no MuSiQue transfer claim. Secondary contrasts descriptive/unadjusted.'))
    with (RUN/'ledger.jsonl').open('a') as f:f.write(json.dumps(dict(event='confirmation_scored',created_unix=time.time(),requests=3200,primary_joint_improvement_confirmed=all(joint.values()),new_analysis_model_calls=0))+'\n')
    print(json.dumps(dict(methods=table,comparisons=comparisons,primary_joint_conditions=joint,selected_cases=len(selection))))


if __name__=='__main__':main()
