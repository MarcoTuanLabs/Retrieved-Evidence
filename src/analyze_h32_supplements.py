"""Verify every captured call, then analyze question-level mechanism contrasts."""
import hashlib,importlib.util,json,sqlite3,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from paired_inference import PairedClusters,interval
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'experiments/h32_mechanism_replication';OLD=ROOT/'experiments/h31_fixed_policy_confirmation'
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def strip(x):
    if isinstance(x,list):return [strip(v) for v in x]
    if isinstance(x,dict):return {k:strip(v) for k,v in x.items() if k not in ['task','_messages','prefix_seconds','prefix_cpu_seconds','ranking_seconds','ranking_cpu_seconds','full_seconds','full_cpu_seconds']}
    return x
def compare(a,b,ids,fields=('f1','em','work_TF')):
    c=PairedClusters(ids,draws=10000,seed=1432);out={'questions':len(ids)}
    for field in fields:
        x=np.array([a[q][field] for q in ids]);y=np.array([b[q][field] for q in ids])
        if field in ['f1','em']:
            d=x-y;out[field]=dict(delta_pp=float(100*d.mean()),paired95_ci_pp=interval(100*c.bootstrap_means(d)),gains=int((d>1e-12).sum()),harms=int((d< -1e-12).sum()))
        else:out[field]=dict(ratio=float(x.mean()/y.mean()),paired95_ratio_ci=interval(c.bootstrap_means(x)/c.bootstrap_means(y)))
    out['margin_sensitivity']={str(m):{f:out[f]['paired95_ci_pp'][0]>=-m for f in ['f1','em']} for m in [.5,1,2,3]}
    return out
def main():
    started=time.time();lock=json.loads((RUN/'FROZEN.json').read_text());cfg=lock['configuration'];cap=RUN/'downloads/capture';payload=RUN/'payload'
    for name,h in lock['hashes'].items():assert sha(ROOT/name)==h,name
    assert sha(OLD/'downloads/capture/records.jsonl')==lock['parent_capture_sha256']
    collection=json.loads((RUN/'COLLECTION.json').read_text());assert collection['verified'] and collection['vault_archive_verified']
    receipt=json.loads((cap/'receipt.json').read_text());assert receipt['state']=='complete' and receipt['completed']==1772
    for name,h in json.loads((cap/'BACKUP.json').read_text())['files'].items():assert sha(cap/name)==h,name
    sys.path[:0]=[str(payload/'native/src'),str(payload),str(ROOT/'src')]
    from transformers import AutoTokenizer
    from costs_v2 import DenseForwardCost
    from h25_replay import Verifier
    from h31_pipeline import execute
    from run_h32_supplements import schedule
    from strategy_capabilities import bounded_messages
    tokenizers={role:AutoTokenizer.from_pretrained(payload/folder,local_files_only=True) for role,folder in [('qwen4b','model4_metadata'),('qwen8b','model8_metadata')]};costs={}
    for role in tokenizers:
        m=json.loads((cap/'runtime'/(role+'.json')).read_text());assert m['revision']==cfg['models'][role]['revision'] and m['model_parameters']==cfg['models'][role]['total_parameters']
        costs[role]=DenseForwardCost.from_config_total(m['model_config'],m['model_parameters'])
    verifier=Verifier(cap/'requests',tokenizers,costs,AutoTokenizer.from_pretrained(payload/'reranker',local_files_only=True),json.loads((payload/'reranker/config.json').read_text()))
    obs={x['id']:x for x in read(payload/'observations.jsonl')};cases={x['id']:x for x in read(payload/'mechanism.jsonl')};records=read(cap/'records.jsonl');tasks=schedule(obs,cases,cfg['seed'])
    assert len(records)==1772 and [r['task'] for r in records]==tasks==json.loads((cap/'schedule.json').read_text())
    ds=cfg['datasets']['twowiki'];db=sqlite3.connect('file:'+str(payload/ds['index'])+'?mode=ro',uri=True)
    for r in records:
        q=r['id'];t=r['task'];assert r['question']==obs[q]['question'] and r['full_seconds']>0 and r['full_cpu_seconds']>=0
        if t['kind']=='baseline':
            calls=iter(r['physical_acquisitions'])
            def search(query,k,excluded):return verifier.retrieval('twowiki',db,query,k,excluded,next(calls),ds['index_sha256'])
            small=SimpleNamespace(tokenizer=tokenizers['qwen4b'],answer=lambda msg:SimpleNamespace(value=verifier.call('qwen4b',msg,r['writer'])),rank=lambda value:SimpleNamespace(value=verifier.rank(value,r['legacy_prefix']['ranking'])))
            big=SimpleNamespace(tokenizer=tokenizers['qwen8b'],answer=lambda msg:SimpleNamespace(value=verifier.call('qwen8b',msg,r['answer'])))
            recreated=execute(obs[q],'bm25_rewrite',search,small,big);assert next(calls,None) is None and strip(recreated)==strip(r),q
        else:
            docs=cases[q]['contexts'][t['context']];assert r['final_documents']==docs
            messages,visibility=bounded_messages(obs[q]['question'],docs,tokenizers['qwen8b']);assert visibility==r['visibility']==r['answer']['visibility']
            answer=verifier.call('qwen8b',messages,r['answer']);assert answer['model_flops']==r['total_model_flops']
    db.close();assert verifier.calls==2572==receipt['model_calls'] and verifier.rankings==800 and verifier.retrievals==1600
    out=RUN/'results';out.mkdir(exist_ok=False)
    verification=dict(status='passed',requests=1772,model_calls=2572,baseline_requests=800,mechanism_requests=972,rankings=verifier.rankings,retrievals=verifier.retrievals,unique_SQL_reexecutions=len(verifier.search_cache),max_BM25_error=verifier.max_score_difference,all_prompt_tokens_and_output_tokens_verified=True,all_costs_recomputed=True,all_schedules_verified=True,gold_in_runtime=False)
    save(out/'replay_verification.json',verification)
    spec=importlib.util.spec_from_file_location('h32_official',ROOT/'data/twowiki_source/official/2wikimultihop_evaluate_v1.1.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    labels={x['id']:x for x in read(ROOT/'data/h31_confirmation/sealed/evaluator.jsonl')}
    def score(q,text):
        refs=[labels[q]['answer']]+labels[q]['answer_aliases'];return dict(f1=max(official.f1_score(text,g)[0] for g in refs),em=float(any(official.exact_match_score(text,g) for g in refs)))
    scored=[]
    for r in records:
        t=r['task'];a=r['answer'];scored.append(dict(id=r['id'],**{k:t[k] for k in ['kind','context','repeat']},**score(r['id'],a['text']),answer=a['text'],work_TF=r['total_model_flops']/1e12,seconds=r['full_seconds'],input_ids_sha256=a['input_ids_sha256'],generated_ids=a['generated_ids'],context_cap=bool(r['visibility'] and (r['visibility']['truncated_document_ids'] or r['visibility']['omitted_document_ids'])),generation_cap=not a['ended_with_eos']))
    (out/'scored.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in scored))
    oldraw={(r['id'],r['method']):r for r in read(OLD/'downloads/capture/records.jsonl')};oldsc={(r['id'],r['method']):r for r in read(OLD/'results/scored.jsonl')}
    baseline={x['id']:x for x in scored if x['kind']=='baseline'};first={q:oldsc[q,'bm25_rewrite'] for q in obs};ids=sorted(obs)
    def mean_table(rows):
        return dict(questions=len(rows),**{f:(None if any(x[f] is None for x in rows.values()) else float(np.mean([x[f] for x in rows.values()]))) for f in ['f1','em','work_TF','seconds']},exact_equivalent=float(sum(x['em'] for x in rows.values())))
    replication=compare(baseline,first,ids);replication.update(first=mean_table(first),repeat=mean_table(baseline),answer_changed=sum(baseline[q]['answer']!=first[q]['answer'] for q in ids),scope='One full independent baseline repeat; same question panel, not 800 new questions; absolute cross-run latency is descriptive because allocated CPU resources may differ.')
    newraw={r['id']:r for r in records if r['task']['kind']=='baseline'}
    replication['writer_input_same']=sum(newraw[q]['writer']['input_ids_sha256']==oldraw[q,'bm25_rewrite']['writer']['input_ids_sha256'] for q in ids)
    replication['writer_text_changed']=sum(newraw[q]['writer']['text']!=oldraw[q,'bm25_rewrite']['writer']['text'] for q in ids)
    replication['final_documents_changed']=sum([d['doc_id'] for d in newraw[q]['final_documents']]!=[d['doc_id'] for d in oldraw[q,'bm25_rewrite']['final_documents']] for q in ids)
    coupled={}
    for name,base in [('H31_original',first),('H32_repeat',baseline)]:
        coupled[name]={}
        for method in ['prefix_only','relation_following']:
            policy={q:(dict(base[q],**score(q,oldraw[q,method]['answer']['text']),work_TF=0) if oldraw[q,method]['accepted_by_rule'] else base[q]) for q in ids}
            coupled[name][method]=dict(comparison=compare(policy,base,ids),exact_count=int(sum(v['em'] for v in policy.values())),f1=float(np.mean([v['f1'] for v in policy.values()])),work_TF=float(np.mean([v['work_TF'] for v in policy.values()])),scope='Offline shared fallback; no newly measured full policy latency, no extra independent sample.')
    contexts=['initial5','rule_context','evidence_documents','rewrite_final10'];values={c:{} for c in contexts};variation={}
    for c in contexts:
        for q in sorted(cases):
            xs=[x for x in scored if x['kind']=='mechanism' and x['id']==q and x['context']==c];assert len(xs)==3
            assert len({x['input_ids_sha256'] for x in xs})==1
            values[c][q]={f:float(np.mean([x[f] for x in xs])) for f in ['f1','em','work_TF','seconds']}
        variation[c]=dict(questions=81,answer_varied=sum(len({x['answer'] for x in scored if x['kind']=='mechanism' and x['id']==q and x['context']==c})>1 for q in cases))
    values['rule']={q:dict(**score(q,oldraw[q,'relation_following']['answer']['text']),work_TF=0.,seconds=None) for q in cases}
    groups=dict(prefix58=sorted(q for q,c in cases.items() if c['prefix_accepted']),added23=sorted(q for q,c in cases.items() if not c['prefix_accepted']),all81=sorted(cases))
    tables={g:{c:mean_table({q:values[c][q] for q in qs}) for c in values} for g,qs in groups.items()}
    pairs=[('prefix58','rule','initial5'),('prefix58','initial5','rewrite_final10'),('added23','rule_context','initial5'),('added23','rule','rule_context'),('all81','evidence_documents','rule_context')]
    contrasts={};selection=[]
    for group,a,b in pairs:
        qs=groups[group];key=group+':'+a+'_minus_'+b;contrasts[key]=compare(values[a],values[b],qs,fields=('f1','em'))
        for sign,label in [(1,'improvement'),(-1,'harm')]:
            found=sorted((q for q in qs if sign*(values[a][q]['em']-values[b][q]['em'])>1e-12),key=lambda q:hashlib.sha256(q.encode()).hexdigest())[:2]
            selection += [dict(id=q,contrast=key,candidate=a,reference=b,direction=label) for q in found]
        if not any(x['contrast']==key for x in selection):selection.append(dict(id=min(qs,key=lambda q:hashlib.sha256(q.encode()).hexdigest()),contrast=key,candidate=a,reference=b,direction='tie'))
    summary=dict(baseline_replication=replication,shared_fallback=coupled,mechanism_tables=tables,mechanism_contrasts=contrasts,within_context_variation=variation,
        requests=1772,model_calls=2572,independent_questions=800,mechanism_questions=81,mechanism_repetitions=3,repeated_rows_counted_as_independent=False,context_caps=sum(x['context_cap'] for x in scored),generation_caps=sum(x['generation_cap'] for x in scored),
        inference='10000 paired question bootstrap draws, seed1432; average repeated readers within question before resampling; all H32 contrasts exploratory/unadjusted, no new independent confirmation.',cost_scope='Mechanism tables contain reader-only work/time. Shared-fallback policies include full baseline neural work but have no new integrated latency measurement.',replay=verification,elapsed_seconds=time.time()-started)
    save(out/'summary.json',summary);save(RUN/'CASE_SELECTION.json',selection)
    review=RUN/'source_review';review.mkdir();exports=[]
    for i,q in enumerate(dict.fromkeys(x['id'] for x in selection),1):
        selected=[x for x in selection if x['id']==q];docs={d['doc_id']:d for context in cases[q]['contexts'].values() for d in context}
        lines=[f'CASE {i}: {q}',obs[q]['question'],'Official reference: '+json.dumps({k:labels[q][k] for k in ['answer','answer_aliases']},ensure_ascii=False),'Contrasts: '+json.dumps(selected,ensure_ascii=False),'Frozen rule answer: '+oldraw[q,'relation_following']['answer']['text']]
        for c in contexts:
            lines += ['',c+' documents: '+json.dumps([d['doc_id'] for d in cases[q]['contexts'][c]]),'Answers: '+json.dumps([dict(answer=x['answer'],f1=x['f1'],em=x['em']) for x in scored if x['kind']=='mechanism' and x['id']==q and x['context']==c],ensure_ascii=False)]
        lines += ['','FULL DOCUMENTS']
        for d in docs.values():lines += ['',d['doc_id']+' '+d['title'],d['body']]
        f=review/f'case_{i:02d}.txt';f.write_text('\n'.join(lines)+'\n');exports.append(dict(id=q,path=str(f.relative_to(RUN)),documents=len(docs),sha256=sha(f)))
    save(RUN/'SOURCE_EXPORTS.json',exports)
    print(json.dumps(dict(status='verified_and_scored',baseline=replication,contrasts=contrasts,review_cases=len(exports)),ensure_ascii=False))
if __name__=='__main__':main()
