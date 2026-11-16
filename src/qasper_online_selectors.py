"""Common input and prompt contract for actual online within-paper selectors."""
import sqlite3
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analyze_qasper_cheap_control import MODES, features
from build_qasper_answerer_pilot import SYSTEM, section_preferences
from freeze_qasper_question_rules import choose
from local_retrieval_server import query_terms

POLICIES=('full','bm25_3','bm25_5','bm25_10','tfidf_5','section_5',
          'question_rule_full','question_rule_top10','learned_0.0005','learned_0.005',
          'semantic_5','semantic_10')


def messages_for(obs, selected):
    chosen=set(selected)
    evidence='\n\n'.join(p['section']+'\n'+p['text'] for p in obs['paragraphs'] if p['id'] in chosen)
    user='Paper: '+obs['title']+'\n\nPaper text:\n'+evidence+'\n\nQuestion: '+obs['question']
    return [{'role':'system','content':SYSTEM},{'role':'user','content':user}]


def fixed_selection(obs, mode):
    paras=obs['paragraphs'];q=obs['question']
    if mode=='full':return [p['id'] for p in paras]
    if mode=='abstract':return ['abstract']
    if mode.startswith('bm25_'):
        con=sqlite3.connect(':memory:')
        try:
            con.execute("CREATE VIRTUAL TABLE chunks USING fts5(pid UNINDEXED, section, text, tokenize='porter unicode61')")
            con.executemany('INSERT INTO chunks VALUES(?,?,?)',[(p['id'],p['section'],p['text']) for p in paras])
            match=' OR '.join('"'+t+'"' for t in query_terms(q))
            rows=con.execute('SELECT pid FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks,0,1,1),pid LIMIT ?',
                             (match,int(mode.split('_')[1]))).fetchall() if match else []
            return [r[0] for r in rows]
        finally:con.close()
    if mode in ('tfidf_5','section_5'):
        vec=TfidfVectorizer(decode_error='replace',strip_accents='unicode',analyzer='word',stop_words='english')
        matrix=vec.fit_transform([p['text'] for p in paras])
        scores=cosine_similarity(matrix,vec.transform([q])).flatten()
        if mode=='section_5':
            priors=section_preferences(q)
            scores=[scores[i]+.15*any(s in p['section'].lower() for s in priors) for i,p in enumerate(paras)]
        ranking=sorted(range(len(paras)),key=lambda i:(-scores[i],paras[i]['id']))
        return [paras[i]['id'] for i in ranking[:5]]
    raise ValueError(mode)


class RecordingTokenizer:
    def __init__(self, tokenizer):self.tokenizer,self.messages=tokenizer,[]
    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        return self.tokenizer.apply_chat_template(messages,**kwargs)


def select(obs, policy, tokenizer, fitted, answerer_cost, ranker=None):
    assert set(obs)=={'question','title','paragraphs'}, 'No evaluator metadata accepted'
    start=time.perf_counter();mode=policy
    detail={'selector_model_flops':0,'selector_kind':policy}
    if policy.startswith('question_rule_'):
        mode=choose(obs['question'])[policy]
    if policy.startswith('learned_'):
        recorder=RecordingTokenizer(tokenizer)
        values,lengths,feature_ms,_=features(obs,recorder)
        pred=np.clip(fitted['model'].predict(values[None,:])[0],0,1)
        costs=np.asarray([answerer_cost.generation(n,o)/1e12 for n,o in zip(lengths,fitted['training_mean_output_lengths'])])
        beta=float(policy.split('_')[1]);selected=int(np.argmax(pred-beta*costs));mode=MODES[selected]
        messages=recorder.messages[selected]
        detail.update(feature_cpu_ms=feature_ms,predicted_f1=pred.tolist(),predicted_answerer_tflops=costs.tolist())
    elif policy.startswith('semantic_'):
        assert ranker is not None
        ranked=ranker.rank(obs['question'],obs['paragraphs'])
        selected=ranked['ranked_paragraph_ids'][:int(policy.split('_')[1])]
        messages=messages_for(obs,selected)
        detail.update(selected_paragraph_ids=selected,ranking=ranked,
                      selector_model_flops=ranked['estimated_linear_attention_flops'])
    else:
        selected=fixed_selection(obs,mode);messages=messages_for(obs,selected)
        detail['selected_paragraph_ids']=selected
    detail.update(chosen_mode=mode,selector_total_seconds=time.perf_counter()-start)
    return messages,detail
