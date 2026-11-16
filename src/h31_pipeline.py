"""Actual independent requests: initial retrieval is executed and paid exactly once."""
import time

from h25_pipeline import prepare_prefix,finish
from h30_relation_acquisition import execute as rule_execute

METHODS=['bm25_direct','bm25_rewrite','prefix_only','relation_following']


def execute(observation,method,search,small,big,synchronize=lambda:None):
    assert set(observation)=={'id','dataset','question'} and method in METHODS
    synchronize();start=time.perf_counter();cpu=time.process_time()
    physical=[]
    def retrieve(query,k,excluded):
        docs,receipt=search(query,k,excluded);physical.append(receipt)
        return docs,receipt
    prefix=rule=initial_receipt=None;accepted=False
    if method=='bm25_direct':
        answer_result=finish(observation,'bm25_10',None,retrieve,big)
    elif method=='bm25_rewrite':
        prefix=prepare_prefix(observation,retrieve,small)
        answer_result=finish(observation,'none',prefix,retrieve,big)
    else:
        initial,initial_receipt=retrieve(observation['question'],5,[])
        clean=[{k:d[k] for k in ['doc_id','title','body']} for d in initial]
        rule=rule_execute(observation['question'],clean,method,retrieve)
        accepted=rule['decision']['accepted']
        if accepted:
            answer_result=dict(answer=dict(text=rule['decision']['answer'],source='explicit_chronology_rule'),
                final_documents=rule['final_documents'],visibility=None,selected_ids=[d['doc_id'] for d in rule['final_documents']],total_model_flops=0.)
        else:
            reused=False
            def fallback_search(query,k,excluded):
                nonlocal reused
                if not reused:
                    assert (query,k,excluded)==(observation['question'],5,[])
                    reused=True
                    return initial,initial_receipt
                return retrieve(query,k,excluded)
            prefix=prepare_prefix(observation,fallback_search,small)
            assert reused and prefix['initial_documents']==initial
            answer_result=finish(observation,'none',prefix,retrieve,big)
    synchronize();wall=time.perf_counter()-start;cput=time.process_time()-cpu
    writer=prefix['writer'] if prefix else None
    total=answer_result['total_model_flops']
    assert total==(0. if accepted else answer_result['answer']['model_flops']+(prefix['prefix_model_flops'] if prefix else 0))
    return dict(**observation,method=method,answer=answer_result['answer'],writer=writer,legacy_prefix=prefix,rule=rule,
        accepted_by_rule=accepted,initial_acquisition=initial_receipt,physical_acquisitions=physical,
        final_documents=answer_result['final_documents'],visibility=answer_result['visibility'],selected_ids=answer_result['selected_ids'],
        total_model_flops=total,full_seconds=wall,full_cpu_seconds=cput,
        timing_scope='Measured complete independent request, CUDA synchronized, includes actual sparse retrieval and rule execution; excludes serialization/backup.')


def self_check():
    import copy
    import hashlib
    import json
    from types import SimpleNamespace
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return [1,2,3]
    class Backend:
        tokenizer=Tokenizer()
        def __init__(self,big=False):self.big=big;self.calls=self.ranks=0
        def answer(self,messages):
            self.calls+=1
            return SimpleNamespace(value=dict(text='synthetic final' if self.big else '{"query":"synthetic bridge"}',model_flops=30 if self.big else 10,request_id=hashlib.sha256(json.dumps(messages).encode()).hexdigest()))
        def rank(self,payload):
            self.ranks+=1
            return SimpleNamespace(value=dict(ranked_paragraph_ids=[d['id'] for d in payload['paragraphs']][::-1],estimated_linear_attention_flops=5))
    docs=[dict(doc_id=str(i),title=f'Doc{i}',body=f'Synthetic passage{i}.',score=float(i)) for i in range(30)]
    def search(q,k,excluded):
        result=[d for d in docs if d['doc_id'] not in excluded][:k]
        return copy.deepcopy(result),dict(query=q,k=k,excluded=excluded,documents=copy.deepcopy(result),wall_seconds=.0,cpu_seconds=.0)
    obs=dict(id='synthetic',dataset='synthetic',question='A question with no comparison?')
    records={m:execute(obs,m,search,Backend(),Backend(True)) for m in METHODS}
    fixed=records['bm25_rewrite']
    for m in ['prefix_only','relation_following']:
        r=records[m]
        assert not r['accepted_by_rule'] and r['total_model_flops']==45
        assert len(r['physical_acquisitions'])==2
        assert r['answer']['_messages']==fixed['answer']['_messages']
        assert r['writer']['_messages']==fixed['writer']['_messages']
        assert r['legacy_prefix']['protected_documents']==fixed['legacy_prefix']['protected_documents']
    people=[dict(doc_id='a',title='Ada Blue',body='Ada Blue (born 1950) is a director.',score=0.),
            dict(doc_id='b',title='Ben Green',body='Ben Green (born 1960) is a director.',score=1.)]
    def resolved(q,k,excluded):return people,dict(query=q,k=k,excluded=excluded,documents=people,wall_seconds=0.,cpu_seconds=0.)
    for m in ['prefix_only','relation_following']:
        small=Backend();big=Backend(True)
        r=execute(dict(obs,question='Who was born earlier, Ada Blue or Ben Green?'),m,resolved,small,big)
        assert r['accepted_by_rule'] and r['answer']['text']=='Ada Blue' and r['total_model_flops']==0
        assert len(r['physical_acquisitions'])==1 and small.calls==small.ranks==big.calls==0
    # Additional acquisitions do not enter the fallback model's prompt.
    unsupported_evidence=dict(obs,question='Which film has the director born earlier, Alpha or Beta?')
    ref=execute(unsupported_evidence,'bm25_rewrite',search,Backend(),Backend(True))
    candidate=execute(unsupported_evidence,'relation_following',search,Backend(),Backend(True))
    assert not candidate['accepted_by_rule'] and len(candidate['physical_acquisitions'])==4
    assert candidate['writer']['_messages']==ref['writer']['_messages'] and candidate['answer']['_messages']==ref['answer']['_messages']
    return dict(status='passed',checks=['fullfallbackpromptparity','initialretrievalexecutedonce','noLLMcallwhenruleaccepts','allactualqueriesretained','newdocumentsneverenterunchangedfallback'],new_model_calls=0)
