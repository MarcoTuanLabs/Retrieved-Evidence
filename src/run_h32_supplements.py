"""Frozen full-baseline replication and evidence-matched reader interventions."""
import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def save(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def schedule(observations,cases,seed):
    tasks=[dict(kind='baseline',id=q,context=None,repeat=0) for q in sorted(observations)]
    tasks += [dict(kind='mechanism',id=q,context=c,repeat=i) for q in sorted(cases) for c in sorted(cases[q]['contexts']) for i in range(3)]
    random.Random(seed).shuffle(tasks)
    return tasks

def main():
    started=time.time();cfg=json.loads((ROOT/'config.json').read_text())
    for name,h in json.loads((ROOT/'manifest.json').read_text())['sha256'].items():assert sha(ROOT/name)==h,name
    out=Path(cfg['output']);out.mkdir(exist_ok=False);(out/'requests').mkdir();(out/'runtime').mkdir()
    for name in ['config.json','manifest.json','CONTRACT.txt','run_h32_supplements.py']:
        shutil.copyfile(ROOT/name,out/'runtime'/name)
    receipt=dict(state='loading',pid=os.getpid(),started_unix=started,completed=0,planned=1772,baseline_completed=0,mechanism_completed=0,model_calls=0,gold_loaded=False)
    def sync():receipt['updated_unix']=time.time();save(out/'receipt.json',receipt)
    sync();sys.path[:0]=[str(ROOT/'native/src'),str(ROOT)];db=None
    try:
        import torch,transformers
        from threadpoolctl import threadpool_limits
        from local_retrieval_server import search
        from shared_retrieval_client import canonical
        from strategy_capabilities import bounded_messages
        from h31_pipeline import execute
        import s4_cuda_backend as implementation
        assert torch.__version__=='2.11.0+cu128' and transformers.__version__=='4.57.1'
        assert 'H100' in torch.cuda.get_device_name()
        receipt['hardware']=dict(name=torch.cuda.get_device_name(),memory=torch.cuda.get_device_properties(0).total_memory)
        torch.set_num_threads(8);torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        backends={}
        for role in ['qwen4b','qwen8b']:
            spec=cfg['models'][role];implementation.MODEL,implementation.REVISION=spec['model'],spec['revision'];implementation.CudaBackend.kind='h32_'+role
            b=implementation.CudaBackend(Path(cfg['model_cache']),ROOT/'reranker');metadata=b.load().value
            assert metadata['model_parameters']==spec['total_parameters'];backends[role]=b;save(out/'runtime'/(role+'.json'),metadata)
            assert time.time()-started<600
        observations={r['id']:r for r in map(json.loads,(ROOT/'observations.jsonl').open())}
        cases={r['id']:r for r in map(json.loads,(ROOT/'mechanism.jsonl').open())}
        assert len(observations)==800 and len(cases)==81 and sum(r['prefix_accepted'] for r in cases.values())==58
        tasks=schedule(observations,cases,cfg['seed']);assert len(tasks)==1772;save(out/'schedule.json',tasks)
        ds=cfg['datasets']['twowiki'];db=sqlite3.connect('file:'+str(ROOT/ds['index'])+'?mode=ro',uri=True)
        def store_call(call):
            messages=call.pop('_messages');body=canonical(messages).encode();key=hashlib.sha256(body).hexdigest()
            assert key==call['request_id'] and call['raw_logits_finite'] and 0<call['generated_tokens']<=128
            p=out/'requests'/(key+'.json')
            if p.exists():assert p.read_bytes()==body
            else:p.write_bytes(body)
            receipt['model_calls']+=1
        receipt.update(state='running',startup_seconds=time.time()-started);sync()
        with torch.inference_mode(),threadpool_limits(limits=1,user_api='blas'),(out/'records.jsonl').open('x') as f:
            for task in tasks:
                assert time.time()<cfg['capture_deadline_unix']-30,'Frozen capture deadline'
                obs=observations[task['id']]
                if task['kind']=='baseline':
                    def retrieve(query,k,excluded):
                        tick=time.perf_counter();cpu=time.process_time();docs,fts=search(db,query,k,excluded)
                        return docs,dict(query=query,k=k,excluded=excluded,documents=docs,fts_query=fts,index_sha256=ds['index_sha256'],wall_seconds=time.perf_counter()-tick,cpu_seconds=time.process_time()-cpu)
                    result=execute(obs,'bm25_rewrite',retrieve,backends['qwen4b'],backends['qwen8b'],torch.cuda.synchronize)
                    store_call(result['writer']);store_call(result['answer']);receipt['baseline_completed']+=1
                else:
                    docs=cases[obs['id']]['contexts'][task['context']]
                    torch.cuda.synchronize();tick=time.perf_counter();cpu=time.process_time()
                    messages,visibility=bounded_messages(obs['question'],docs,backends['qwen8b'].tokenizer)
                    call=backends['qwen8b'].answer(messages).value;torch.cuda.synchronize()
                    call.update(role='answer8',visibility=visibility,_messages=messages)
                    result=dict(**obs,answer=call,visibility=visibility,final_documents=docs,total_model_flops=call['model_flops'],
                        full_seconds=time.perf_counter()-tick,full_cpu_seconds=time.process_time()-cpu,timing_scope='reader-only including prompt construction; acquisition excluded')
                    store_call(result['answer']);receipt['mechanism_completed']+=1
                result['task']=task;f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();receipt['completed']+=1
                if receipt['completed']%20==0:sync();print(json.dumps({k:receipt[k] for k in ['completed','baseline_completed','mechanism_completed','model_calls']}),flush=True)
                if receipt['completed']%100==0:os.fsync(f.fileno())
            os.fsync(f.fileno())
        assert receipt['baseline_completed']==800 and receipt['mechanism_completed']==972 and receipt['model_calls']==2572
        receipt.update(state='complete',finished_unix=time.time(),worker_seconds=time.time()-started);sync()
    except BaseException as e:
        receipt.update(state='failed',error=repr(e),traceback=traceback.format_exc(),finished_unix=time.time());sync();raise
    finally:
        if db is not None:db.close()
        save(out/'BACKUP.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},scope='local capture manifest; persistent archive verified at collection'))

if __name__=='__main__':main()
