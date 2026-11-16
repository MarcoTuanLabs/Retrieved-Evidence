"""Score and verify the frozen draft-removal comparison; never retune the candidate."""
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
import numpy as np
from analyze_validation import scorers
from paired_inference import PairedClusters, interval

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'experiments/h19_draft_free_baseline'


def read(p): return [json.loads(line) for line in p.read_text().splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    started = time.perf_counter()
    lock = json.loads((RUN/'FROZEN.json').read_text())
    for path, expected in lock['hashes'].items(): assert sha(ROOT/path) == expected, path
    assert json.loads((RUN/'COLLECTION.json').read_text())['verified']
    assert json.loads((RUN/'RESOURCE_STOP.json').read_text())['state'] == 'stopped'
    cap = RUN/'downloads/capture'
    receipt = json.loads((cap/'receipt.json').read_text())
    assert receipt['state'] == 'complete' and receipt['completed_requests'] == receipt['planned_requests'] == 400
    assert receipt['gold_loaded'] is False and receipt['model_outcome_reuse'] is False and receipt['kv_reuse'] is False
    for path, expected in json.loads((cap/'BACKUP.json').read_text())['files'].items(): assert sha(cap/path) == expected, path
    out = RUN/'results'; out.mkdir(exist_ok=False)
    payload = RUN/'payload'; cfg = json.loads((payload/'config.json').read_text())
    sys.path[:0] = [str(payload/'native/src'), str(payload), str(ROOT/'src')]
    from h19_pipeline import METHODS
    from h19_replay import Replay, EXACT_FIELDS
    from costs_v2 import DenseForwardCost
    from transformers import AutoTokenizer
    tokenizers = {role:AutoTokenizer.from_pretrained(payload/folder, local_files_only=True)
        for role,folder in [('qwen4b','model4_metadata'),('qwen8b','model8_metadata')]}
    costs = {}
    for role in tokenizers:
        meta = receipt[role]['metadata']; spec = cfg['models'][role]
        assert meta['model'] == spec['model'] and meta['revision'] == spec['revision']
        assert meta['model_parameters'] == spec['total_parameters']
        assert receipt[role]['engineering']['state'] == 'passed'
        costs[role] = DenseForwardCost.from_config_total(meta['model_config'], meta['model_parameters'])
        assert costs[role].__dict__ == meta['cost_model']
    replay = Replay(cap/'requests', tokenizers, costs,
        AutoTokenizer.from_pretrained(payload/'reranker',local_files_only=True),
        json.loads((payload/'reranker/config.json').read_text()))
    observations = {r['id']:r for r in read(payload/'observations.jsonl')}
    labels = {r['id']:r for r in read(ROOT/'data/fresh_v3/sealed/selection_evaluator.jsonl')}
    assert len(observations) == 200 and set(observations) == set(labels)
    rows = read(cap/'musique_development_outputs.jsonl')
    schedule = json.loads((cap/'musique_development_schedule.json').read_text())
    assert len(rows) == len(schedule) == len({(r['id'],r['method']) for r in rows}) == 400
    db = sqlite3.connect('file:'+str(payload/'index.sqlite')+'?mode=ro',uri=True)
    score, scorer = scorers('h3_confirmation'); scored = []
    for row, entry in zip(rows,schedule):
        assert entry == dict(id=row['id'],method=row['method'])
        assert row['dataset'] == 'musique_development' and row['method'] in METHODS
        result = replay.run(observations[row['id']],row['method'],row,db,'musique_development')
        for key in EXACT_FIELDS: assert result[key] == row[key], (row['id'],key)
        assert row['pipeline_seconds'] > 0
        gold = labels[row['id']]
        f1, em = score(row['answer'], [gold['answer']]+gold['answer_aliases'])
        scored.append(dict(id=row['id'],method=row['method'],f1=f1,em=float(em),prediction=row['answer'],
            work_TF=row['total_model_flops']/1e12,seconds=row['pipeline_seconds'],
            process_cpu_seconds=row['process_cpu_seconds'],draft_TF=row['draft_model_flops']/1e12,
            hops=gold['hops'],source_group=gold['source_group']))
    db.close()
    assert replay.calls == 1000
    by = {(r['id'],r['method']):r for r in rows}
    pairs = []
    for q in sorted(observations):
        paid, no = (by[q,m] for m in METHODS)
        checks = {k:paid[k] == no[k] for k in ['initial_documents','acquired_documents','protected_documents',
            'final_documents','writer_parse','selected_ids','visibility','prefix_model_flops','ranker_model_flops']}
        checks.update(writer_input_same=paid['calls'][0]['input_ids_sha256']==no['calls'][0]['input_ids_sha256'],
            writer_tokens_same=paid['calls'][0]['generated_ids']==no['calls'][0]['generated_ids'],
            final_messages_same=paid['calls'][-1]['request_id']==no['calls'][-1]['request_id'],
            final_input_tokens_same=paid['calls'][-1]['input_ids_sha256']==no['calls'][-1]['input_ids_sha256'],
            final_output_tokens_same=paid['calls'][-1]['generated_ids']==no['calls'][-1]['generated_ids'],
            final_answer_same=paid['answer']==no['answer'],
            ranking_decisions_same=(paid['ranking'] or {}).get('ranked_paragraph_ids')==(no['ranking'] or {}).get('ranked_paragraph_ids'),
            exactly_draft_work_removed=paid['total_model_flops']-no['total_model_flops']==paid['draft_model_flops'],
            strictly_less_model_work=no['total_model_flops']<paid['total_model_flops'])
        pairs.append(dict(id=q,checks=checks,all_equal_and_less_work=all(checks.values())))
    mismatches = [p for p in pairs if not p['all_equal_and_less_work']]
    ids = sorted(observations); indexed = {(r['id'],r['method']):r for r in scored}
    arrays = {m:{k:np.array([indexed[q,m][k] for q in ids]) for k in ['f1','em','work_TF','seconds']} for m in METHODS}
    clusters = PairedClusters([labels[q]['source_group'] for q in ids]); assert len(clusters.members)==186
    contrast = {}
    for key in ('f1','em'):
        d = arrays[METHODS[1]][key]-arrays[METHODS[0]][key]
        ci = [100*x for x in interval(clusters.bootstrap_means(d))]
        contrast[key] = dict(delta_pp=100*float(d.mean()),source95_ci_pp=ci,
                            margins_pp={str(m):ci[0]>=-m for m in (.5,1,2,3)})
    for key in ('work_TF','seconds'):
        a,b = arrays[METHODS[1]][key],arrays[METHODS[0]][key]
        contrast[key] = dict(ratio=float(a.mean()/b.mean()),source95_ratio_ci=interval(clusters.bootstrap_means(a)/clusters.bootstrap_means(b)))
    table = [dict(method=m,**{k:float(v.mean()) for k,v in arrays[m].items()},
        p95_seconds=float(np.quantile(arrays[m]['seconds'],.95))) for m in METHODS]
    subgroups = {str(h):{m:{k:float(np.mean([r[k] for r in scored if r['method']==m and r['hops']==h]))
        for k in ('f1','em','work_TF','seconds')} for m in METHODS} for h in sorted({r['hops'] for r in labels.values()})}
    summary = dict(status='scored_pending_disposition',questions=200,source_groups=186,table=table,
        candidate_minus_reference=contrast,p95_ratio=table[1]['p95_seconds']/table[0]['p95_seconds'],
        paired_exact_checks_passed=200-len(mismatches),mismatches=len(mismatches),subgroups=subgroups,
        preliminary_disposition='keep_equivalent_lower_compute_baseline' if not mismatches else 'no_equivalence_claim_pending_mismatch_review',
        replayed_model_calls=replay.calls,ranker_windows_costs_reconstructed=replay.rankings,
        ranker_inference_reexecuted=False,unique_retrievals=len(replay.search_cache),max_retrieval_difference=replay.max_search_difference,
        official_scorer=scorer,wall_seconds=time.perf_counter()-started,
        scope='Exposed200 development; one locked intervention; paired source bootstrap; no independent quality confirmation or novel mechanism claim')
    save(out/'summary.json',summary);save(out/'pair_checks.json',pairs)
    (out/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    (out/'mismatch_packets.jsonl').write_text(''.join(json.dumps(dict(**p,question=observations[p['id']]['question'],
        reference=labels[p['id']]['answer'],records={m:by[p['id'],m] for m in METHODS}),ensure_ascii=False)+'\n' for p in mismatches))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('subgroups','official_scorer')},ensure_ascii=False))


if __name__ == '__main__': main()
