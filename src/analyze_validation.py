"""Unchanged official labels, paired source-cluster gates, and complete reporting."""
import argparse,ast,collections,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from paired_inference import PairedClusters,interval
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def scorers(run):
    if run=='h3_confirmation':
        base=ROOT.parent/'musique_pilot200_20261114/literature/official';sys.path.insert(0,str(base))
        from metrics.answer import compute_f1,compute_exact
        return lambda answer,refs:(max(compute_f1(r,answer) for r in refs),max(compute_exact(r,answer) for r in refs)),dict(path=str(base/'metrics/answer.py'),sha256=sha(base/'metrics/answer.py'))
    path=PROJECT/'epistemics_provenance/controller_v3/experiments/controller_rescue_audit_v1/reference_hotpotqa_36358534/hotpot_evaluate_v1.py'
    # Execute unchanged official metric function AST; omit unrelated file I/O
    # and ujson/pickle dependencies from the official command-line evaluator.
    module=ast.parse(path.read_text());nodes=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('normalize_answer','f1_score','exact_match_score')]
    assert len(nodes)==3
    import re,string
    scope=dict(re=re,string=string,Counter=collections.Counter)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    f,e=scope['f1_score'],scope['exact_match_score']
    assert f('yes','yes')[0]==1 and f('yes indeed','yes')[0]==0 and e('The United States','United States')
    return lambda answer,refs:(max(f(answer,r)[0] for r in refs),max(e(answer,r) for r in refs)),dict(path=str(path),sha256=sha(path),official_functions_unchanged=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',choices=['h3_confirmation','h4_transfer'],required=True)
    a=ap.parse_args();run=ROOT/'experiments'/a.run;out=run/'results';cap=run/'downloads/capture'
    lock=json.loads((run/'FROZEN.json').read_text());assert sha(run/'PROTOCOL.txt')==lock['protocol_sha256']
    replay=json.loads((out/'replay_verification.json').read_text());assert replay['status']=='passed'
    config=json.loads((run/'payload/run_config.json').read_text());methods=lock['methods'];assert methods==config['methods']
    if a.run=='h3_confirmation':
        data=ROOT/'data/fresh_v3';labelname='sealed/confirmation_evaluator.jsonl';metadata='fresh_data_receipt.json';n=600
    else:
        data=ROOT/'data/hotpot_transfer_v1';labelname='sealed/evaluator.jsonl';metadata='hotpot_data_receipt.json';n=400
    fingerprint=json.loads((ROOT/'experiments'/metadata).read_text())['fingerprints'][labelname]
    assert sha(data/labelname)==fingerprint
    labels={r['id']:r for r in read(data/labelname)};ids=sorted(labels);assert len(ids)==n
    rows=read(cap/'outputs.jsonl');assert len(rows)==config['planned_requests']
    rows=[r for r in rows if r['phase']=='main'];by={(r['id'],r['method'],r['repetition']):r for r in rows}
    assert set(by)==set(itertools.product(ids,methods,[0]))
    score,scorer=scorers(a.run);records=[];routes={m:collections.Counter() for m in methods}
    for r in rows:
        e=labels[r['id']];d=r['details'];s=set(e['support_ids']);visible=set(d['selected_document_ids']);acquired=set(d['acquired_document_ids'])
        f1,em=score(r['answer'],[e['answer']]+e['answer_aliases']);decision=r.get('controller_decision',{})
        total=sum(c['model_flops'] for c in r['calls'])+d['selector_model_flops'];assert total==r['total_model_flops']
        routes[r['method']][decision.get('selected_method',str(decision.get('continue',decision.get('continue_search',decision.get('stop','fixed')))))]+=1
        last=r['calls'][-1]
        records.append(dict(id=r['id'],method=r['method'],f1=f1,em=float(em),work_TF=total/1e12,seconds=r['pipeline_seconds'],
            selected=len(visible),support_visible=len(s&visible)/len(s),support_acquired=len(s&acquired)/len(s),support_complete=int(s<=visible),
            answer_cap=int(last['generated_tokens']==128 and not last['ended_with_eos']),
            any_context_truncation=int(any(c['visibility']['truncated_document_ids'] or c['visibility']['omitted_document_ids'] for c in r['calls'])),
            model_calls=len(r['calls']),search_calls=len(r['acquisition_calls']),controller_ms=1000*decision.get('wall_seconds',decision.get('rule_seconds',0)),
            controller_TF=d.get('controller_model_flops',0)/1e12,selector_TF=d['selector_model_flops']/1e12))
    sb={(r['id'],r['method']):r for r in records};fields=[k for k in records[0] if k not in ('id','method')]
    values={m:{f:np.asarray([[sb[q,m][f]] for q in ids]) for f in fields} for m in methods}
    cluster=PairedClusters([labels[q]['source_group'] for q in ids]);table={}
    for m,v in values.items():
        table[m]={f:float(arr.mean()) for f,arr in v.items()}
        table[m].update(p95_ms=float(np.quantile(v['seconds'],.95)*1000),f1_source95_ci=interval(cluster.bootstrap_means(v['f1'][:,0])),action_counts=dict(routes[m]))
    reference=lock['reference'];primary=lock['primary_candidate'];assert reference in methods and primary in methods
    comparisons={m:cluster.compare(values[m],values[reference]) for m in methods}
    for m,c in comparisons.items():
        delta=values[m]['f1'][:,0]-values[reference]['f1'][:,0]
        c['question_outcomes']=dict(better=int((delta>1e-12).sum()),tie=int((abs(delta)<=1e-12).sum()),worse=int((delta< -1e-12).sum()),
            exact_matches_gained=int(((values[m]['em'][:,0]==1)&(values[reference]['em'][:,0]==0)).sum()),
            exact_matches_lost=int(((values[m]['em'][:,0]==0)&(values[reference]['em'][:,0]==1)).sum()),
            positive_f1_mass_pp=float(np.maximum(delta,0).mean()*100),negative_f1_mass_pp=float(np.minimum(delta,0).mean()*100))
        leave_one_group=[float((delta.sum()-delta[ix].sum())/(n-len(ix))*100) for ix in cluster.members]
        c['leave_one_source_group_out_delta_pp_range']=[min(leave_one_group),max(leave_one_group)]
    pairs=[(primary,b) for b in ['local_query','local_initial','local_state_mean_mask','query_full',lock['best_rule']]]
    pairs += [('binary_answer','binary_support'),('binary_answer','binary_retrospective')]
    secondary={x+' minus '+y:cluster.compare(values[x],values[y]) for x,y in pairs}
    strata={}
    categories=sorted(set(str(labels[q]['hops']) if a.run=='h3_confirmation' else labels[q]['type'] for q in ids))
    for category in categories:
        mask=np.asarray([(str(labels[q]['hops']) if a.run=='h3_confirmation' else labels[q]['type'])==category for q in ids])
        strata[category]=dict(questions=int(mask.sum()),methods={m:{f:float(v[f][mask].mean()) for f in ['f1','em','work_TF','seconds','support_visible']} for m,v in values.items()})
    local=lock['local_fixed_methods'];f=np.stack([values[m]['f1'][:,0] for m in local],axis=1)
    oracle=dict(methods=local,f1=float(f.max(1).mean()),best_local_fixed_f1=float(f.mean(0).max()),
        headroom_pp=float((f.max(1).mean()-f.mean(0).max())*100),questions_with_differing_f1=int((np.ptp(f,axis=1)>1e-12).sum()),
        scope='Finite reachable-path hindsight only; future labels, free decisions; no learnability or runtime claim')
    result=dict(run=a.run,questions=n,source_groups=len(cluster.members),methods=methods,reference=reference,primary_candidate=primary,best_rule=lock['best_rule'],
        primary_result=comparisons[primary],table=table,comparisons_to_reference=comparisons,secondary_comparisons=secondary,strata=strata,
        local_oracle=oracle,scorer=scorer,evaluator_sha256=fingerprint,lock_sha256=sha(run/'FROZEN.json'),capture_sha256=sha(cap/'outputs.jsonl'),
        terminal_http_errors=replay['terminal_http_errors'],all_questions_retained=True,source_review_exclusions=0,
        selection_or_retuning_on_validation=False,model_work_scope='Modeled padded dense/attention operations, not hardware FLOP or energy counters',
        timing_scope='One fresh full request per method/question; within this GPU/process only; source-cluster uncertainty',
        multiplicity_scope='One frozen primary. All remaining contrasts descriptive; no uncorrected additional confirmatory claims.')
    save(out/'summary.json',result);(out/'scored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    lines=[('MuSiQue 600 題獨立確認' if a.run=='h3_confirmation' else 'Hotpot 400 題零調參移轉'),'','預先固定主方法：'+primary+'；選模集決定的固定基準：'+reference,
        '全部問題保留。每題每策略完整執行一次；時間只在本輪比較。','',
        '方法 | F1% | EM% | 平均 TF | 平均 ms | p95 ms | 相對基準 F1差與95%來源群組區間 | 全部門檻']
    for m in methods:
        t=table[m];c=comparisons[m];d=c['comparisons']['f1'];lo,hi=d['paired95_ci_pp']
        lines.append(f"{m} | {100*t['f1']:.3f} | {100*t['em']:.3f} | {t['work_TF']:.3f} | {1000*t['seconds']:.1f} | {t['p95_ms']:.1f} | {d['delta_pp']:+.3f} [{lo:+.3f},{hi:+.3f}]pp | {c['all_gates_pass']}")
    c=result['primary_result'];lines+=['','預先固定主方法門檻：',json.dumps(c['gates'],ensure_ascii=False),
        '計算量與平均時間門檻採點估計；品質門檻使用配對95%來源群組區間下界。',
        '門檻：F1與EM下界各>=-2pp、計算量至少省25%、平均時間至少省10%、p95時間增幅不超過10%。',
        '逐題品質變化：'+json.dumps(c['question_outcomes'],ensure_ascii=False),
        '平均品質非劣不表示每一題都不受損；逐題得失與來源群組影響另列。',
        '次要比較未作多重比較校正；不能挑事後最佳方法取代預先固定主方法。','',
        '次要比較（前者減後者）：']
    for name,c in secondary.items():
        d=c['comparisons']['f1'];lo,hi=d['paired95_ci_pp']
        lines.append(f"{name}: F1 {d['delta_pp']:+.3f}pp [{lo:+.3f},{hi:+.3f}]；計算量節省{c['comparisons']['work_TF']['reduction_percent']:+.2f}%；平均時間節省{c['comparisons']['seconds']['reduction_percent']:+.2f}%")
    lines+=['','限制：單一4B回答模型、既定檢索器與pooled語料；MuSiQue抽樣有hop分布變化；Hotpot不是full-wiki標準測試。',
        '計算量為公式估算，不等於實測硬體FLOP或耗電。官方答案指標不代表每條來源鏈均已人工驗證。']
    (out/'RESULTS_ZH.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(run=a.run,primary=result['primary_result'],table=table),indent=2))
if __name__=='__main__':main()
