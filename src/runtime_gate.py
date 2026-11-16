"""Synthetic engineering only, with expected literals unrelated to dataset labels."""
import hashlib,json,re,time
from strategy_capabilities import bounded_messages
from reliable_capture import atomic_json
CASES=[
 ('Project Lantern has the code VELA.','What is the code of Project Lantern?','VELA'),
 ('In this record, the town of Zefton has the mayor Nila.','Who is the mayor of Zefton in this record?','Nila'),
 ('The archive labels item seven as quartz.','What is item seven labeled?','quartz'),
 ('The scheduled launch year of mission Cedar is 2042.','What is the scheduled launch year of mission Cedar?','2042'),
 ('The measured length of cable Y is 632 meters.','How many meters long is cable Y?','632'),
 ('The dossier identifies the destination as Orven.','What destination does the dossier identify?','Orven'),
 ('The cargo container has a violet lid.','What color is the cargo container lid?','violet'),
 ('The signature word on the receipt is PELTA.','What is the signature word on the receipt?','PELTA'),
]
def normalized(s):return re.sub(r'[^a-z0-9]','',s.casefold())
def run_gate(backend,out):
    start=time.time();records=[];generate=backend.generate
    for mode in ['compiled','eager']:
        if mode=='eager':
            def eager(x,capacity,**kw):return generate(x,capacity,disable_compile=True,**kw)
            backend.generate=eager
        try:
            for i,(body,q,expected) in enumerate(CASES):
                docs=[dict(doc_id=hashlib.sha256(body.encode()).hexdigest(),title='Synthetic engineering record',body=body,score=0.)]
                messages,visibility=bounded_messages(q,docs,backend.tokenizer)
                value=backend.answer(messages).value
                records.append(dict(mode=mode,index=i,expected=expected,messages=messages,call=value,
                                    correct=normalized(value['text'])==normalized(expected)))
                atomic_json(out/'runtime/g0_capture.json',dict(state='running',records=records))
        finally:backend.generate=generate
    by={(r['mode'],r['index']):r for r in records}
    assert all(r['correct'] for r in records),'Synthetic literal-answer gate failed; preserve raw capture'
    assert all(by['compiled',i]['call']['input_ids_sha256']==by['eager',i]['call']['input_ids_sha256'] for i in range(8))
    result=dict(state='passed',cases=8,model_calls=16,all_correct=True,
        exact_cross_mode_tokens=sum(by['compiled',i]['call']['generated_ids']==by['eager',i]['call']['generated_ids'] for i in range(8)),
        elapsed_seconds=time.time()-start,records=records)
    atomic_json(out/'runtime/g0_capture.json',result)
    # Exercise compiled path again, clearing reusable static cache before return.
    import torch
    token=backend.tokenizer.encode('example',add_special_tokens=False)[0]
    with torch.inference_mode():
        x=torch.full((1,32),token,device='cuda',dtype=torch.long)
        for capacity in backend.caches:backend.generate(x,capacity,max_new_tokens=2,min_new_tokens=2)
        for cache in backend.caches.values():cache.reset()
    return {k:v for k,v in result.items() if k!='records'}
