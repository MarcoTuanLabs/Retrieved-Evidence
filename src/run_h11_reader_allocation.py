"""One fresh reader process over three immutable captured evidence contexts."""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def engineering_gate(backend, out):
    from runtime_gate import CASES, normalized
    from strategy_capabilities import bounded_messages
    started = time.time()
    generate = backend.generate
    records = []
    for mode in ('compiled', 'eager'):
        if mode == 'eager':
            def eager(x, capacity, **kw):
                return generate(x, capacity, disable_compile=True, **kw)
            backend.generate = eager
        try:
            for index, (body, question, expected) in enumerate(CASES):
                docs = [dict(doc_id=hashlib.sha256(body.encode()).hexdigest(), title='Synthetic engineering record', body=body, score=0.)]
                messages, _ = bounded_messages(question, docs, backend.tokenizer)
                call = backend.answer(messages).value
                accepted = expected if isinstance(expected, list) else [expected]
                records.append(dict(mode=mode, index=index, expected=expected, messages=messages,
                                    call=call, literal_correct=normalized(call['text']) in [normalized(x) for x in accepted]))
                atomic(out/'runtime/g0_capture.json', dict(state='running', records=records))
        finally:
            backend.generate = generate
    by = {(x['mode'], x['index']): x for x in records}
    for i in range(len(CASES)):
        a, b = by['compiled', i]['call'], by['eager', i]['call']
        assert a['input_ids_sha256'] == b['input_ids_sha256']
        assert a['generated_ids'] == b['generated_ids'], 'Synthetic compiled/eager token mismatch'
        assert a['raw_logits_finite'] and b['raw_logits_finite']
    original = dict(backend.common)
    try:
        backend.common.update(max_new_tokens=128, min_new_tokens=128)
        boundary = backend.answer(records[0]['messages']).value
        assert boundary['generated_tokens'] == 128 and boundary['raw_logits_finite']
    finally:
        backend.common = original
    result = dict(state='passed', cases=len(CASES), model_calls=len(records)+1,
                  exact_cross_mode_tokens=len(CASES), literal_correct=sum(x['literal_correct'] for x in records),
                  literal_accuracy_is_diagnostic_not_quality_gate=True,
                  records=records, boundary_probe=boundary, elapsed_seconds=time.time()-started)
    atomic(out/'runtime/g0_capture.json', result)
    # Restore compiled warm state after eager probes, as in prior frozen backend.
    import torch
    token = backend.tokenizer.encode('example', add_special_tokens=False)[0]
    with torch.inference_mode():
        x = torch.full((1, 32), token, device='cuda', dtype=torch.long)
        for capacity in backend.caches:
            backend.generate(x, capacity, max_new_tokens=2, min_new_tokens=2)
        for cache in backend.caches.values():
            cache.reset()
    return {k:v for k,v in result.items() if k not in ('records','boundary_probe')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader', choices=['qwen4b','qwen8b'], required=True)
    args = parser.parse_args()
    started = time.time()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert sha(ROOT/name) == expected, name
    cfg = json.loads((ROOT/'config.json').read_text())
    model = cfg['models'][args.reader]
    out = Path(cfg['output'])/args.reader
    vault = Path(cfg['vault_output'])/args.reader
    out.mkdir(parents=True, exist_ok=False)
    vault.mkdir(parents=True, exist_ok=False)
    for base in (out, vault):
        for name in ('runtime', 'requests'):
            (base/name).mkdir()
    for name in ('manifest.json','config.json','CONTRACT.txt','run_h11_reader_allocation.py'):
        shutil.copy2(ROOT/name, out/'runtime'/name)
    receipt = dict(state='loading', reader=args.reader, model=model, started_unix=started,
                   completed=0, planned=600, prefix_reuse=True, full_online_run=False, gold_loaded=False)
    def save():
        atomic(out/'receipt.json', receipt)
        atomic(vault/'receipt.json', receipt)
    save()
    sys.path[:0] = [str(ROOT/'native/src'), str(ROOT)]
    try:
        import torch
        import transformers
        from threadpoolctl import threadpool_limits
        import s4_cuda_backend as implementation
        from shared_retrieval_client import canonical
        assert torch.__version__ == '2.11.0+cu128' and transformers.__version__ == '4.57.1'
        assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 5090'
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Model selection is explicit; underlying backend and cost formula stay frozen.
        implementation.MODEL = model['model']
        implementation.REVISION = model['revision']
        implementation.CudaBackend.kind = 'actual_qwen_dense_bf16_sdpa_static_compiled_' + args.reader
        backend = implementation.CudaBackend(Path(cfg['model_cache']), ROOT/'reranker')
        metadata = backend.load().value
        assert metadata['model_parameters'] == model['total_parameters']
        atomic(out/'runtime/backend.json', metadata)
        receipt['g0'] = engineering_gate(backend, out)
        save()
        rows = [json.loads(x) for x in (ROOT/'contexts.jsonl').read_text().splitlines()]
        expected_inputs = json.loads((ROOT/'expected_inputs.json').read_text())[args.reader]
        assert len(rows) == 600
        by = {(x['id'],x['policy']):x for x in rows}
        lengths = []
        for row in rows:
            assert [d['doc_id'] for d in row['documents']] == row['selected_ids']
            messages = row['messages']
            body = canonical(messages).encode()
            rid = hashlib.sha256(body).hexdigest()
            assert rid == row['baseline_request_id']
            ids = backend.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            info = expected_inputs[row['id']+'|'+row['policy']]
            assert len(ids) == info['tokens'] <= 15360
            assert hashlib.sha256(canonical(ids).encode()).hexdigest() == info['sha256']
            if args.reader == 'qwen4b':
                assert info['sha256'] == row['baseline_input_ids_sha256']
            lengths.append(len(ids))
            for base in (out, vault):
                request = base/'requests'/(rid+'.json')
                if request.exists():
                    assert request.read_bytes() == body
                else:
                    request.write_bytes(body)
        atomic(out/'runtime/input_preflight.json', dict(inputs=600, max_prompt_tokens=max(lengths), all_model_specific_input_hashes_verified=True, no_evidence_truncation=True))
        rng = random.Random(cfg['seed'])
        ids = sorted({x['id'] for x in rows})
        rng.shuffle(ids)
        schedule = []
        for q in ids:
            policies = list(cfg['policies'])
            rng.shuffle(policies)
            schedule.extend(dict(id=q, policy=p) for p in policies)
        atomic(out/'schedule.json', schedule)
        receipt.update(state='running', startup_seconds=time.time()-started)
        save()
        with torch.inference_mode(), threadpool_limits(limits=1, user_api='blas'), (out/'outputs.jsonl').open('x') as f, (vault/'outputs.jsonl').open('x') as v:
            for entry in schedule:
                assert time.time()-started < cfg['max_reader_seconds']-120, 'Frozen reader deadline reached'
                row = by[entry['id'],entry['policy']]
                tick = time.perf_counter()
                answer = backend.answer(row['messages']).value
                elapsed = time.perf_counter()-tick
                assert answer['request_id'] == row['baseline_request_id']
                result = dict(**entry, reader=args.reader, answer=answer, selected_ids=row['selected_ids'],
                              prefix_model_flops=row['prefix_model_flops'], total_model_flops=row['prefix_model_flops']+answer['model_flops'],
                              answer_stage_seconds=elapsed, timing_scope='Fresh answer stage; inherited prefix arithmetic only')
                line = json.dumps(result, ensure_ascii=False)+'\n'
                f.write(line); f.flush(); v.write(line); v.flush()
                receipt['completed'] += 1
                if receipt['completed'] % 25 == 0:
                    receipt.update(updated_unix=time.time())
                    save()
                    print(json.dumps(dict(reader=args.reader,completed=receipt['completed'],planned=600)), flush=True)
            os.fsync(f.fileno()); os.fsync(v.fileno())
        receipt.update(state='complete', finished_unix=time.time())
        save()
    except BaseException as exc:
        receipt.update(state='failed', error=repr(exc), traceback=traceback.format_exc(), finished_unix=time.time())
        save()
        raise
    finally:
        shutil.copytree(out/'runtime', vault/'runtime', dirs_exist_ok=True)
        if (out/'schedule.json').exists():
            shutil.copy2(out/'schedule.json', vault/'schedule.json')
        files = {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
        for name, expected in files.items():
            assert (vault/name).is_file() and sha(vault/name) == expected, name
        atomic(out/'BACKUP.json',dict(verified=True,files=files,vault=str(vault),created_unix=time.time()))
        shutil.copy2(out/'BACKUP.json',vault/'BACKUP.json')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
