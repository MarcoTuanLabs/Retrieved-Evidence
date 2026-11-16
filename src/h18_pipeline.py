"""Full causal repair pipeline. Inputs contain no evaluator or future outcomes."""
import time
from h18_runtime_policy import choose,mixture_action
from strategy_capabilities import bounded_messages,parse_writer

METHODS=['primary','fixed_stop4','fixed_upgrade8','fixed_original_query','fixed_cached20','trained_mix4']

def execute(observation,method,search,backend4,backend8,artifact,mix):
    assert set(observation)=={'id','question','dataset'} and method in METHODS
    q=observation['question'];calls=[];acquisitions=[];events=[]
    def acquire(query,k,excluded):
        docs,receipt=search(query,k,excluded);acquisitions.append(receipt)
        events.append(dict(event='search',number=len(acquisitions),query=query,documents=[d['doc_id'] for d in docs]))
        return docs
    def call(backend,documents,role,history=None):
        messages,visibility=bounded_messages(q,documents,backend.tokenizer,history,False)
        value=backend.answer(messages).value
        value.update(role=role,visibility=visibility,_messages=messages)
        calls.append(value);events.append(dict(event=role,call=len(calls),request_id=value['request_id']))
        return value
    initial=acquire(q,5,[])
    writer=call(backend4,initial,'query_writer',[q]);query,parse=parse_writer(writer['text'],q,False)
    new=acquire(query,min(15,20-len(initial)),[d['doc_id'] for d in initial])
    assert not {d['doc_id'] for d in initial}&{d['doc_id'] for d in new}
    ranking=None;rank_cost=0;kept=[]
    if new:
        ranking=backend4.rank(dict(question=query,paragraphs=[dict(id=d['doc_id'],section=d['title'],text=d['body']) for d in new])).value
        order=ranking['ranked_paragraph_ids'];assert set(order)=={d['doc_id'] for d in new}
        keep=set(order[:5]);kept=[d for d in new if d['doc_id'] in keep]
        rank_cost=ranking['estimated_linear_attention_flops'];events.append(dict(event='ranking',selected=[d['doc_id'] for d in kept]))
    protected=initial+kept;pool=initial+new
    draft=call(backend4,protected,'draft4')
    decision_observation=None;decision_seconds=0;head=0
    if method=='primary':
        tick=time.perf_counter()
        if len(protected)!=10:
            route=dict(action='cached20',predicted_gains=None,head_flops=0,fallback='out_of_domain_short_context')
        else:
            decision_observation=dict(question=q,documents=[{k:d[k] for k in ['title','body']} for d in protected],
                draft={k:draft[k] for k in ['text','generated_tokens','ended_with_eos']})
            route=choose(decision_observation,artifact)
        action=route['action'];head=route['head_flops'];decision_seconds=time.perf_counter()-tick
    elif method=='trained_mix4':
        tick=time.perf_counter();action=mixture_action(observation['dataset'],observation['id'],mix['selection']['weights'])
        route=dict(action=action,randomization=mix['randomization']);decision_seconds=time.perf_counter()-tick
    else:
        action=method.removeprefix('fixed_');route=dict(action=action)
    events.append(dict(event='decision',action=action,model_calls_so_far=len(calls),searches_so_far=len(acquisitions)))
    if action=='stop4':answer=draft;final_documents=protected
    else:
        final_documents=pool if action=='cached20' else protected
        if action=='original_query':final_documents=protected+acquire(q,5,[d['doc_id'] for d in pool])
        assert action in ['upgrade8','original_query','cached20']
        answer=call(backend8,final_documents,'answer8')
    assert events[[e['event'] for e in events].index('decision')]['model_calls_so_far']==2
    return dict(answer=answer['text'],action=action,calls=calls,acquisition_calls=acquisitions,events=events,
        initial_documents=initial,acquired_documents=pool,protected_documents=protected,final_documents=final_documents,
        ranking=ranking,writer_parse=parse,route=route,decision_observation=decision_observation,decision_seconds=decision_seconds,
        prefix_model_flops=writer['model_flops']+rank_cost,draft_model_flops=draft['model_flops'],ranker_model_flops=rank_cost,head_flops=head,
        total_model_flops=rank_cost+sum(c['model_flops'] for c in calls)+head,
        selected_ids=answer['visibility']['visible_document_ids'],visibility=answer['visibility'])

def self_check(artifact,mix):
    """Cheap synthetic causal-order and full-cost fixtures, no research answers."""
    import hashlib,json
    from types import SimpleNamespace
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return [1,2,3]
    docs=[dict(doc_id=hashlib.sha256(str(i).encode()).hexdigest(),title=f'Doc {i}',body=f'Synthetic evidence number {i}.',score=float(i)) for i in range(25)]
    class Backend:
        tokenizer=Tokenizer()
        def __init__(self,big=False):self.n=0;self.big=big
        def answer(self,messages):
            self.n+=1;text='Synthetic final' if self.big else ('{"query":"synthetic bridge"}' if self.n==1 else 'Synthetic draft')
            return SimpleNamespace(value=dict(text=text,model_flops=30 if self.big else 10,generated_tokens=3,ended_with_eos=True,request_id=hashlib.sha256(json.dumps(messages).encode()).hexdigest()))
        def rank(self,payload):
            return SimpleNamespace(value=dict(ranked_paragraph_ids=[d['id'] for d in payload['paragraphs']],estimated_linear_attention_flops=5))
    checked=[]
    for method in METHODS:
        b4=Backend();b8=Backend(True)
        def search(q,k,excluded):return [d for d in docs if d['doc_id'] not in excluded][:k],dict(query=q,k=k,excluded=excluded)
        r=execute(dict(id='synthetic',question='What is the synthetic answer?',dataset='synthetic'),method,search,b4,b8,artifact,mix)
        assert b4.n==2 and b8.n==int(r['action']!='stop4')
        assert r['total_model_flops']==25+30*(r['action']!='stop4')+r['head_flops']
        events=[e['event'] for e in r['events']];assert events.index('draft4')<events.index('decision')
        if 'answer8'in events:assert events.index('decision')<events.index('answer8')
        assert len(r['protected_documents'])==10 and len(r['acquired_documents'])==20
        checked.append(method)
    b4=Backend();b8=Backend(True)
    def short(q,k,excluded):return [d for d in docs[:3] if d['doc_id'] not in excluded][:k],dict(query=q,k=k,excluded=excluded)
    r=execute(dict(id='short',question='Synthetic?',dataset='synthetic'),'primary',short,b4,b8,artifact,mix)
    assert r['action']=='cached20' and r['route']['fallback']=='out_of_domain_short_context' and r['head_flops']==0
    return dict(status='passed',synthetic_policies=checked,short_context_fallback=True,causal_action_before_future_calls=True,full_cost_fixtures=True)
