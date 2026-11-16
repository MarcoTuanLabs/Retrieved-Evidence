"""Verify requests, retrieval identities, ranker windows and independent model work."""
import copy
import hashlib
import math
from types import SimpleNamespace

class Verifier:
    def __init__(self,requests,tokenizers,costs,ranker_tokenizer,ranker_config):
        self.requests=requests;self.tokenizers=tokenizers;self.costs=costs
        self.rt=ranker_tokenizer;self.rc=ranker_config;self.search_cache={}
        self.calls=0;self.rankings=0;self.retrievals=0;self.max_score_difference=0.

    def call(self,role,messages,captured):
        from shared_retrieval_client import canonical
        a=copy.deepcopy(captured);tok=self.tokenizers[role];cost=self.costs[role]
        body=canonical(messages).encode();rid=hashlib.sha256(body).hexdigest();assert rid==a['request_id']
        assert (self.requests/(rid+'.json')).read_bytes()==body
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
        assert len(ids)==a['prompt_tokens']<=15360
        assert hashlib.sha256(canonical(ids).encode()).hexdigest()==a['input_ids_sha256']
        assert tok.decode(a['generated_ids'],skip_special_tokens=True).strip()==a['text']
        assert len(a['generated_ids'])==a['generated_tokens'] and 0<a['generated_tokens']<=128
        assert a['ended_with_eos']==(a['generated_ids'][-1]==tok.eos_token_id)
        assert a['raw_logits_finite'] and a['raw_logits_checked_steps']==a['generated_tokens']
        n=len(ids);extra=max(0,a['generated_tokens']-1)
        capacity=next(c for c in [1024,2048,4096,8192,16384] if c>=n+128);assert capacity==a['static_capacity']
        work=2*cost.transformer_parameters*n+2*cost.head_parameters+2*(cost.transformer_parameters+cost.head_parameters)*extra+4*cost.layers*cost.attention_width*capacity*(n+extra)
        assert work==a['model_flops'];self.calls+=1
        return a

    def rank(self,payload,record):
        from semantic_paragraph_baseline import SemanticParagraphRanker
        r=copy.deepcopy(record);ids=[p['id'] for p in payload['paragraphs']]
        assert len(set(r['ranked_paragraph_ids']))==len(ids) and set(r['ranked_paragraph_ids'])==set(ids)
        pairs,windows,details=SemanticParagraphRanker.windows(SimpleNamespace(tokenizer=self.rt),payload['question'],payload['paragraphs'])
        shapes=[[len(pairs[o:o+16]),max(len(v['input_ids']) for v in pairs[o:o+16])] for o in range(0,len(pairs),16)]
        assert windows==r['windows'] and details==r['details'] and shapes==r['batch_shapes']
        assert len(windows)==len(r['window_scores']) and all(math.isfinite(s) for s in r['window_scores'])
        scores={q:-math.inf for q in ids}
        for w,s in zip(windows,r['window_scores']):scores[w['paragraph_id']]=max(scores[w['paragraph_id']],s)
        assert scores==r['paragraph_scores'] and sorted(scores,key=lambda q:(-scores[q],q))==r['ranked_paragraph_ids']
        hidden,intermediate,layers=(self.rc[k] for k in ['hidden_size','intermediate_size','num_hidden_layers'])
        work=sum(2*b*l*layers*(4*hidden**2+2*hidden*intermediate)+2*b*(hidden**2+hidden*len(self.rc['id2label']))+4*layers*b*l*l*hidden for b,l in shapes)
        assert work==r['estimated_linear_attention_flops'] and sum(b*l for b,l in shapes)==r['padded_tokens']
        self.rankings+=1;return r

    def retrieval(self,dataset,db,query,k,excluded,record,index_sha256):
        from local_retrieval_server import search
        assert record['query']==query and record['k']==k and record['excluded']==excluded and record['index_sha256']==index_sha256
        key=dataset,query,k,tuple(excluded)
        if key not in self.search_cache:self.search_cache[key]=search(db,query,k,excluded)
        docs,fts=self.search_cache[key];assert fts==record['fts_query'] and len(docs)==len(record['documents'])
        for actual,old in zip(docs,record['documents']):
            assert {a:b for a,b in actual.items() if a!='score'}=={a:b for a,b in old.items() if a!='score'}
            diff=abs(actual['score']-old['score']);assert diff<=1e-12;self.max_score_difference=max(self.max_score_difference,diff)
        assert record['wall_seconds']>=0 and record['cpu_seconds']>=0
        self.retrievals+=1;return copy.deepcopy(record['documents']),copy.deepcopy(record)
