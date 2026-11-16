"""Paired draft-removal execution on exposed development questions; no worker labels."""
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
from run_h11_reader_allocation import atomic,sha,engineering_gate

ROOT=Path(__file__).resolve().parent

def main():
    started=time.time();cfg=json.loads((ROOT/'config.json').read_text());manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,h in manifest['sha256'].items():assert sha(ROOT/name)==h,name
    out=Path(cfg['output']);vault=Path(cfg['vault_output'])
    for p in (out,vault):p.mkdir(parents=True,exist_ok=False)
    (out/'runtime').mkdir();(out/'requests').mkdir();(vault/'runtime').mkdir();(vault/'requests').mkdir()
    for name in ['config.json','manifest.json','CONTRACT.txt','run_h19_baseline.py','h19_pipeline.py','h18_pipeline.py','h18_runtime_policy.py','h13_signal_features.py']:
        for p in (out,vault):shutil.copy2(ROOT/name,p/'runtime'/name)
    receipt=dict(state='loading',pid=os.getpid(),started_unix=started,completed_requests=0,planned_requests=cfg['planned_requests'],panels={},gold_loaded=False,full_online_run=True,model_outcome_reuse=False,kv_reuse=False,exposed_development=True)
    def save():
        for p in (out,vault):atomic(p/'receipt.json',receipt)
    save();connections={}
    sys.path[:0]=[str(ROOT/'native/src'),str(ROOT)]
    try:
        import torch,transformers,numpy as np
        from threadpoolctl import threadpool_limits
        import s4_cuda_backend as implementation
        from shared_retrieval_client import canonical
        from local_retrieval_server import search
        from h19_pipeline import METHODS,execute,self_check
        assert torch.__version__=='2.11.0+cu128' and transformers.__version__=='4.57.1'
        gpu=torch.cuda.get_device_name();assert 'RTX PRO 6000' in gpu and torch.cuda.get_device_properties(0).total_memory>=90*1024**3,gpu
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        receipt['synthetic_pipeline']=self_check();save();backends={}
        for role in ['qwen4b','qwen8b']:
            assert time.time()-started<cfg['max_startup_seconds'],'Startup/engineering deadline'
            spec=cfg['models'][role];implementation.MODEL=spec['model'];implementation.REVISION=spec['revision']
            implementation.CudaBackend.kind='h19_dual_resident_pinned_'+role
            backend=implementation.CudaBackend(Path(cfg['model_cache']),ROOT/'reranker');meta=backend.load().value
            assert meta['model_parameters']==spec['total_parameters'];backends[role]=backend
            gate=out/role;gate.mkdir();(gate/'runtime').mkdir()
            receipt[role]=dict(metadata=meta,engineering=engineering_gate(backend,gate));save()
            assert time.time()-started<cfg['max_startup_seconds'],'Startup/engineering deadline'
        rows=[json.loads(l) for l in (ROOT/'observations.jsonl').read_text().splitlines()]
        assert len(rows)==cfg['questions'] and all(set(r)=={'id','question','dataset'} for r in rows)
        parity=json.loads((ROOT/'retrieval_parity.json').read_text())
        for dataset,spec in cfg['datasets'].items():
            db=sqlite3.connect('file:'+str(ROOT/spec['index'])+'?mode=ro',uri=True);connections[dataset]=db
            for r in rows:
                if r['dataset']!=dataset:continue
                ds,fts=search(db,r['question'],5,[]);assert [d['doc_id'] for d in ds]==parity[dataset+'|'+r['id']]
        receipt.update(state='running',startup_seconds=time.time()-started,both_models_resident=True);save()
        for dataset in cfg['dataset_order']:
            panel_start=time.time();selected=[r for r in rows if r['dataset']==dataset];random.Random(cfg["seed"]).shuffle(selected)
            schedule=[]
            for r in selected:
                order=METHODS.copy();random.Random(str(cfg['seed'])+'|'+dataset+'|'+r['id']).shuffle(order)
                schedule.extend(dict(id=r['id'],method=m) for m in order)
            atomic(out/(dataset+'_schedule.json'),schedule);byid={r['id']:r for r in selected}
            panel=dict(state='running',started_unix=panel_start,completed=0,planned=len(schedule));receipt['panels'][dataset]=panel;save()
            with torch.inference_mode(),threadpool_limits(limits=1,user_api='blas'),(out/(dataset+'_outputs.jsonl')).open('x') as f,(vault/(dataset+'_outputs.jsonl')).open('x') as v:
                def retrieve(query,k,excluded):
                    tick,cpu=time.perf_counter(),time.process_time();docs,fts=search(connections[dataset],query,k,excluded)
                    rec=dict(query=query,k=k,excluded=excluded,fts_query=fts,documents=docs,index_sha256=cfg['datasets'][dataset]['index_sha256'],wall_seconds=time.perf_counter()-tick,cpu_seconds=time.process_time()-cpu)
                    return docs,rec
                for i,entry in enumerate(schedule):
                    assert time.time()-panel_start<cfg['max_panel_seconds']-120,'Frozen panel deadline'
                    observation=byid[entry['id']];tick,cpu=time.perf_counter(),time.process_time()
                    result=execute(observation,entry['method'],retrieve,backends['qwen4b'],backends['qwen8b'])
                    json.dumps(dict(answer=result['answer'],selected_ids=result['selected_ids']))
                    elapsed=time.perf_counter()-tick;cpu_elapsed=time.process_time()-cpu
                    for call in result['calls']:
                        messages=call.pop('_messages');body=canonical(messages).encode();assert hashlib.sha256(body).hexdigest()==call['request_id']
                        for folder in (out,vault):
                            path=folder/'requests'/(call['request_id']+'.json')
                            if path.exists():assert path.read_bytes()==body
                            else:path.write_bytes(body)
                    result.update(dataset=dataset,**entry,request_index=i,pipeline_seconds=elapsed,process_cpu_seconds=cpu_elapsed)
                    line=json.dumps(result,ensure_ascii=False)+'\n'
                    for sink in (f,v):sink.write(line);sink.flush()
                    panel['completed']=i+1;receipt['completed_requests']+=1
                    if (i+1)%25==0:
                        receipt['updated_unix']=time.time();save();print(json.dumps(dict(dataset=dataset,completed=i+1,planned=len(schedule))),flush=True)
                    if (i+1)%100==0:
                        for sink in (f,v):os.fsync(sink.fileno())
                for sink in (f,v):os.fsync(sink.fileno())
            panel.update(state='complete',finished_unix=time.time());save()
            shutil.copy2(out/(dataset+'_schedule.json'),vault/(dataset+'_schedule.json'))
        assert receipt['completed_requests']==cfg['planned_requests']
        receipt.update(state='complete',finished_unix=time.time());save()
    except BaseException as error:
        receipt.update(state='failed',error=repr(error),traceback=traceback.format_exc(),finished_unix=time.time());save();raise
    finally:
        for db in connections.values():db.close()
        for p in out.rglob('*'):
            if p.is_file():
                rel=p.relative_to(out);target=vault/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists() or sha(p)!=sha(target):shutil.copyfile(p,target)
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
        for name,h in hashes.items():assert sha(vault/name)==h,name
        backup=dict(verified=True,files=hashes,vault=str(vault),created_unix=time.time())
        atomic(out/'BACKUP.json',backup);atomic(vault/'BACKUP.json',backup)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['qwen4b','qwen8b']}),flush=True)

if __name__=='__main__':main()
