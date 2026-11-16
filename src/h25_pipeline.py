"""Five natural retrieval endpoints; shared prefix is fully charged per policy."""
import time
from strategy_capabilities import bounded_messages,parse_writer

METHODS=['none','top1','top5','cached20','bm25_10']

def prepare_prefix(observation,search,backend4):
    assert set(observation)=={'id','question','dataset'}
    started=time.perf_counter();cpu=time.process_time();q=observation['question']
    initial,a=search(q,5,[])
    messages,visibility=bounded_messages(q,initial,backend4.tokenizer,[q],False)
    writer=backend4.answer(messages).value
    writer.update(role='query_writer',visibility=visibility,_messages=messages)
    query,parse=parse_writer(writer['text'],q,False)
    added,b=search(query,min(15,20-len(initial)),[d['doc_id'] for d in initial])
    assert not {d['doc_id'] for d in initial}&{d['doc_id'] for d in added}
    ranking=None;rank_cost=0;kept=[];ranking_seconds=0.;ranking_cpu=0.
    if added:
        tick=time.perf_counter();c=time.process_time()
        ranking=backend4.rank(dict(question=query,paragraphs=[dict(id=d['doc_id'],section=d['title'],text=d['body']) for d in added])).value
        ranking_seconds=time.perf_counter()-tick;ranking_cpu=time.process_time()-c
        order=ranking['ranked_paragraph_ids'];assert set(order)=={d['doc_id'] for d in added}
        keep=set(order[:5]);kept=[d for d in added if d['doc_id'] in keep]
        rank_cost=ranking['estimated_linear_attention_flops']
    return dict(id=observation['id'],dataset=observation['dataset'],question=q,
        initial_documents=initial,acquired_documents=initial+added,protected_documents=initial+kept,
        writer=writer,writer_query=query,writer_parse=parse,ranking=ranking,acquisition_calls=[a,b],
        writer_model_flops=writer['model_flops'],ranker_model_flops=rank_cost,
        prefix_model_flops=writer['model_flops']+rank_cost,ranking_seconds=ranking_seconds,ranking_cpu_seconds=ranking_cpu,
        prefix_seconds=time.perf_counter()-started,prefix_cpu_seconds=time.process_time()-cpu)

def finish(observation,method,prefix,search,backend8):
    assert set(observation)=={'id','question','dataset'} and method in METHODS
    started=time.perf_counter();cpu=time.process_time();q=observation['question'];acq=[]
    if method=='bm25_10':
        assert prefix is None,'Direct baseline must not consume the writer prefix'
        docs,receipt=search(q,10,[]);acq.append(receipt)
        prefix_work=0;prefix_seconds=0.;prefix_cpu=0.
    else:
        assert prefix['id']==observation['id'] and prefix['dataset']==observation['dataset'] and prefix['question']==q
        prefix_work=prefix['prefix_model_flops'];prefix_seconds=prefix['prefix_seconds'];prefix_cpu=prefix['prefix_cpu_seconds']
        docs=list(prefix['acquired_documents'] if method=='cached20' else prefix['protected_documents'])
        if method in ['top1','top5']:
            excluded=[d['doc_id'] for d in prefix['acquired_documents']]
            new,receipt=search(q,1 if method=='top1' else 5,excluded);acq.append(receipt)
            assert not {d['doc_id'] for d in new}&set(excluded)
            docs+=new
    messages,visibility=bounded_messages(q,docs,backend8.tokenizer)
    answer=backend8.answer(messages).value
    answer.update(role='answer8',visibility=visibility,_messages=messages)
    suffix_seconds=time.perf_counter()-started;suffix_cpu=time.process_time()-cpu
    return dict(id=observation['id'],dataset=observation['dataset'],method=method,answer=answer,
        final_documents=docs,acquisition_calls=acq,selected_ids=visibility['visible_document_ids'],visibility=visibility,
        prefix_model_flops=prefix_work,answer_model_flops=answer['model_flops'],total_model_flops=prefix_work+answer['model_flops'],
        suffix_seconds=suffix_seconds,suffix_cpu_seconds=suffix_cpu,
        attributed_full_seconds=prefix_seconds+suffix_seconds,attributed_full_cpu_seconds=prefix_cpu+suffix_cpu,
        timing_scope='direct_complete_request' if method=='bm25_10' else 'shared_measured_prefix_plus_fresh_suffix')

def self_check():
    """Guard scientific contrasts: existing baseline parity, no free prefix, no leakage."""
    import copy
    import hashlib
    import json
    from types import SimpleNamespace
    from h19_pipeline import execute as established
    class Tokenizer:
        def apply_chat_template(self,*a,**kw):return [1,2,3]
    class Backend:
        tokenizer=Tokenizer()
        def __init__(self,big=False):self.big=big;self.calls=0;self.ranks=0
        def answer(self,messages):
            self.calls+=1
            return SimpleNamespace(value=dict(text='synthetic final' if self.big else '{"query":"synthetic bridge"}',
                model_flops=30 if self.big else 10,request_id=hashlib.sha256(json.dumps(messages).encode()).hexdigest()))
        def rank(self,payload):
            self.ranks+=1
            return SimpleNamespace(value=dict(ranked_paragraph_ids=[d['id'] for d in payload['paragraphs']][::-1],estimated_linear_attention_flops=5))
    checked=[]
    for count in [0,3,5,8,20,25]:
        docs=[dict(doc_id=str(i),title=f'Doc{i}',body=f'Synthetic passage{i}.',score=float(i)) for i in range(count)]
        calls=[]
        def search(q,k,excluded):
            result=[d for d in docs if d['doc_id'] not in excluded][:k]
            r=dict(query=q,k=k,excluded=excluded,documents=copy.deepcopy(result));calls.append(r)
            return copy.deepcopy(result),r
        obs=dict(id='synthetic',dataset='synthetic',question='Synthetic question?')
        b4=Backend();b8=Backend(True);p=prepare_prefix(obs,search,b4);snapshot=copy.deepcopy(p)
        result={m:finish(obs,m,None if m=='bm25_10' else p,search,b8) for m in METHODS}
        assert p==snapshot and b4.calls==1 and b8.calls==5
        reference=established(obs,'no_draft_fixed10',search,Backend(),Backend(True))
        assert p['writer']['_messages']==reference['calls'][0]['_messages']
        for k in ['initial_documents','acquired_documents','protected_documents','writer_parse','ranking']:
            assert p[k]==reference[k],k
        assert result['none']['answer']['_messages']==reference['calls'][-1]['_messages']
        assert result['none']['total_model_flops']==reference['total_model_flops']
        assert all(result[m]['total_model_flops']==p['prefix_model_flops']+30 for m in METHODS if m!='bm25_10')
        assert result['bm25_10']['total_model_flops']==30 and result['bm25_10']['prefix_model_flops']==0
        assert result['cached20']['final_documents']==p['acquired_documents'] and not result['cached20']['acquisition_calls']
        assert not result['none']['acquisition_calls']
        for m,k in [('top1',1),('top5',5)]:
            r=result[m]['acquisition_calls'][0];assert r['k']==k and r['query']==obs['question']
            assert r['excluded']==[d['doc_id'] for d in p['acquired_documents']]
            assert len(result[m]['final_documents'])==len(p['protected_documents'])+min(k,max(0,count-20))
        assert [d['doc_id'] for d in result['bm25_10']['final_documents']]==[str(i) for i in range(min(count,10))]
        assert all(set(o['answer']['_messages'][-1])=={'role','content'} for o in result.values())
        checked.append(dict(available=count,protected=len(p['protected_documents']),acquired=len(p['acquired_documents']),
            established_baseline_parity=True,shared_prefix_unchanged=True,fully_charged_prefix=True,direct_baseline_no_writer_cost=True))
    return dict(status='passed',fixtures=checked,model_calls=0,learned_decisions=0)
