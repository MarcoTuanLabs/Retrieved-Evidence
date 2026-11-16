"""Resume only missing independent requests from the same frozen schedule; no gold in payload."""
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


def main():
    started=time.time();cfg=json.loads((ROOT/'config.json').read_text())
    for name,h in json.loads((ROOT/'manifest.json').read_text())['sha256'].items():assert sha(ROOT/name)==h,name
    out=Path(cfg['output']);vault=Path(cfg['vault_output'])
    from h31_resume_state import validate_prefix
    prior,done,prior_schedule=validate_prefix(cfg['resume_from'],cfg['resume_mirror'])
    oldcfg=json.loads((Path(cfg['resume_from'])/'runtime/config.json').read_text())
    assert all(cfg[k]==v for k,v in oldcfg.items() if k!='capture_deadline_unix')
    for folder in [out,vault]:
        shutil.copytree(cfg['resume_from'],folder)
        (folder/'BACKUP.json').rename(folder/'PHASE0_BACKUP.json')
        shutil.copyfile(folder/'receipt.json',folder/'PHASE0_RECEIPT.json')
        for name in ['config.json','qwen4b.json','qwen8b.json']:
            shutil.copyfile(folder/'runtime'/name,folder/'runtime'/('phase0_'+name))
        shutil.copyfile(ROOT/'run_h31_resume.py',folder/'runtime/run_h31_resume.py')
        shutil.copyfile(ROOT/'h31_resume_state.py',folder/'runtime/h31_resume_state.py')
        for name in ['config.json','manifest.json','CONTRACT.txt','run_h31_confirmation.py','h31_pipeline.py','h30_relation_acquisition.py','h29_comparator.py','h25_pipeline.py']:
            shutil.copyfile(ROOT/name,folder/'runtime'/name)
    receipt=dict(state='loading',pid=os.getpid(),started_unix=started,completed=0,planned=cfg['questions']*4,
        completed_questions=0,planned_questions=cfg['questions'],completed_writers=0,completed_readers=0,
        completed_rule_answers=0,completed_rankings=0,physical_sparse_queries=0,
        gold_loaded=False,independent_full_requests=True,campaign_main_model_flops=0)
    for key in ['completed','completed_questions','completed_writers','completed_readers','completed_rule_answers','completed_rankings','physical_sparse_queries','campaign_main_model_flops']:
        receipt[key]=prior[key]
    receipt.update(resumed=True,phase0_completed=prior['completed'],phase0_started_unix=prior['started_unix'],
        phase0_finished_unix=prior['finished_unix'],phase0_startup_seconds=prior['startup_seconds'],
        phase0_records_sha256=sha(Path(cfg['resume_from'])/'records.jsonl'),actual_model_load_blocks=2)
    def sync_receipt():
        receipt['updated_unix']=time.time()
        for folder in [out,vault]:save(folder/'receipt.json',receipt)
    sync_receipt();dbs={};sys.path[:0]=[str(ROOT/'native/src'),str(ROOT)]
    try:
        import torch,transformers
        from threadpoolctl import threadpool_limits
        from local_retrieval_server import search
        from shared_retrieval_client import canonical
        from h31_pipeline import METHODS,execute
        import s4_cuda_backend as implementation
        assert torch.__version__=='2.11.0+cu128' and transformers.__version__=='4.57.1'
        assert 'H100' in torch.cuda.get_device_name() and torch.cuda.get_device_properties(0).total_memory>=75*1024**3
        receipt['hardware'] = dict(name=torch.cuda.get_device_name(), total_memory=torch.cuda.get_device_properties(0).total_memory)
        assert METHODS==cfg['methods']
        torch.set_num_threads(8);torch.manual_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        backends={}
        for role in ['qwen4b','qwen8b']:
            spec=cfg['models'][role];implementation.MODEL,implementation.REVISION=spec['model'],spec['revision']
            implementation.CudaBackend.kind='h31_frozen_'+role
            backend=implementation.CudaBackend(Path(cfg['model_cache']),ROOT/'reranker');metadata=backend.load().value
            assert metadata['model_parameters']==spec['total_parameters'];backends[role]=backend
            for folder in [out,vault]:save(folder/'runtime'/(role+'.json'),metadata)
            assert time.time()-started<cfg['startup_seconds']
        observations=[json.loads(l) for l in (ROOT/'observations.jsonl').read_text().splitlines()]
        by={(r['dataset'],r['id']):r for r in observations};assert len(by)==len(observations)==cfg['questions']
        rng=random.Random(cfg['seed']);keys=sorted(by);rng.shuffle(keys);schedule=[]
        for ds,q in keys:
            methods=list(METHODS);rng.shuffle(methods);schedule.append(dict(dataset=ds,id=q,methods=methods))
        assert schedule==prior_schedule
        for folder in [out,vault]:save(folder/'schedule.json',schedule)
        for ds,spec in cfg['datasets'].items():dbs[ds]=sqlite3.connect('file:'+str(ROOT/spec['index'])+'?mode=ro',uri=True)
        def store_call(call):
            if call is None or call.get('source')=='explicit_chronology_rule':return
            messages=call.pop('_messages');body=canonical(messages).encode();request=hashlib.sha256(body).hexdigest()
            assert request==call['request_id'] and call['raw_logits_finite'] and 0<call['generated_tokens']<=128
            for folder in [out,vault]:
                p=folder/'requests'/(request+'.json')
                if p.exists():assert p.read_bytes()==body
                else:p.write_bytes(body)
        receipt.update(state='running',startup_seconds=time.time()-started,total_startup_seconds=prior['startup_seconds']+time.time()-started);sync_receipt()
        with torch.inference_mode(),threadpool_limits(limits=1,user_api='blas'),(out/'records.jsonl').open('a') as f,(vault/'records.jsonl').open('a') as vf:
            for entry in schedule:
                ds,q=entry['dataset'],entry['id'];obs=by[ds,q]
                if all((ds,q,m) in done for m in entry['methods']):continue
                for method in entry['methods']:
                    if (ds,q,method) in done:continue
                    assert time.time()-started<cfg['worker_seconds']-120,'Frozen worker deadline'
                    assert time.time()<cfg['capture_deadline_unix']-120,'User wallclock deadline; retain partial capture'
                    def retrieve(query,k,excluded):
                        t=time.perf_counter();cpu=time.process_time();docs,fts=search(dbs[ds],query,k,excluded)
                        return docs,dict(query=query,k=k,excluded=excluded,documents=docs,fts_query=fts,index_sha256=cfg['datasets'][ds]['index_sha256'],
                            wall_seconds=time.perf_counter()-t,cpu_seconds=time.process_time()-cpu)
                    result=execute(obs,method,retrieve,backends['qwen4b'],backends['qwen8b'],torch.cuda.synchronize)
                    store_call(result['writer']);store_call(result['answer'])
                    line=json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n'
                    for sink in [f,vf]:sink.write(line);sink.flush()
                    receipt['completed']+=1;receipt['completed_writers']+=int(result['writer'] is not None)
                    receipt['completed_rule_answers']+=int(result['accepted_by_rule']);receipt['completed_readers']+=int(not result['accepted_by_rule'])
                    receipt['completed_rankings']+=int(result['legacy_prefix'] is not None and result['legacy_prefix']['ranking'] is not None)
                    receipt['physical_sparse_queries']+=len(result['physical_acquisitions']);receipt['campaign_main_model_flops']+=result['total_model_flops']
                receipt['completed_questions']+=1
                if receipt['completed_questions']%10==0:
                    sync_receipt();print(json.dumps({k:receipt[k] for k in ['completed','planned','completed_questions','completed_writers','completed_readers','completed_rule_answers']}),flush=True)
                if receipt['completed_questions']%50==0:
                    for sink in [f,vf]:os.fsync(sink.fileno())
            for sink in [f,vf]:os.fsync(sink.fileno())
        assert receipt['completed']==cfg['questions']*4 and receipt['completed_questions']==cfg['questions']
        assert receipt['completed_readers']+receipt['completed_rule_answers']==receipt['completed']
        receipt.update(state='complete',finished_unix=time.time(),worker_seconds=time.time()-started,combined_worker_seconds=prior['finished_unix']-prior['started_unix']+time.time()-started,campaign_wall_seconds=time.time()-prior['started_unix']);sync_receipt()
    except BaseException as error:
        receipt.update(state='failed',error=repr(error),traceback=traceback.format_exc(),finished_unix=time.time());sync_receipt();raise
    finally:
        for db in dbs.values():db.close()
        hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
        for name,h in hashes.items():
            p=vault/name;p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists() or sha(p)!=h:shutil.copyfile(out/name,p)
            assert sha(p)==h
        for folder in [out,vault]:save(folder/'BACKUP.json',dict(verified=True,files=hashes,vault=str(vault),created_unix=time.time()))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
