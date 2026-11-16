"""Leakage-separated cheap learned-Q diagnostic on already-open training papers."""
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer

from build_qasper_answerer_pilot import SYSTEM, section_preferences
from costs_v2 import DenseForwardCost
from local_retrieval_server import query_terms

ROOT = Path(__file__).resolve().parents[1]
MODES = ('abstract', 'bm25_3', 'bm25_5', 'bm25_10', 'tfidf_5', 'section_5', 'full')
BETAS = (0, .0005, .001, .002, .005, .01, .02)


def observation(row):
    # Project away evaluator fields before the feature function sees this row.
    return {key: row[key] for key in ('question', 'title', 'paragraphs')}


def features(obs, tok):
    start = time.perf_counter()
    q, paras = obs['question'], obs['paragraphs']
    texts = [p['text'] for p in paras]
    vec = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word', stop_words='english')
    matrix = vec.fit_transform(texts)
    scores = cosine_similarity(matrix, vec.transform([q])).flatten()
    # Rebuild and query the small paper index here. Its CPU time is charged,
    # rather than assuming an earlier saved ranking is free at deployment.
    con = sqlite3.connect(':memory:')
    con.execute("CREATE VIRTUAL TABLE chunks USING fts5(pid UNINDEXED, section, text, tokenize='porter unicode61')")
    con.executemany('INSERT INTO chunks VALUES(?,?,?)', [(p['id'], p['section'], p['text']) for p in paras])
    match = ' OR '.join('"' + t + '"' for t in query_terms(q))
    bm25 = [r[0] for r in con.execute('SELECT pid FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks,0,1,1),pid', (match,))] if match else []
    con.close()
    rank = sorted(range(len(paras)), key=lambda i: (-scores[i], paras[i]['id']))
    priors = section_preferences(q)
    section_scores = [scores[i] + .15 * any(s in paras[i]['section'].lower() for s in priors) for i in range(len(paras))]
    srank = sorted(range(len(paras)), key=lambda i: (-section_scores[i], paras[i]['id']))
    selections = {'abstract': ['abstract'], 'full': [p['id'] for p in paras],
        **{'bm25_' + str(k): bm25[:k] for k in (3, 5, 10)},
        'tfidf_5': [paras[i]['id'] for i in rank[:5]], 'section_5': [paras[i]['id'] for i in srank[:5]]}
    lengths = []
    for mode in MODES:
        chosen = set(selections[mode])
        evidence = '\n\n'.join(p['section'] + '\n' + p['text'] for p in paras if p['id'] in chosen)
        user = 'Paper: ' + obs['title'] + '\n\nPaper text:\n' + evidence + '\n\nQuestion: ' + q
        messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
        lengths.append(len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)))
    sorted_scores = np.sort(scores)[::-1]
    mass = scores / max(float(scores.sum()), 1e-12)
    entropy = -float(np.sum(mass * np.log(np.maximum(mass, 1e-12)))) / np.log(max(len(paras), 2))
    terms = set(vec.build_analyzer()(q))
    toptext = ' '.join(paras[i]['text'] for i in rank[:5])
    coverage = len(terms & set(vec.build_analyzer()(toptext))) / max(len(terms), 1)
    a, b = set(selections['bm25_5']), set(selections['tfidf_5'])
    values = [len(q.split()), len(obs['title'].split()), len(paras), len(set(p['section'] for p in paras)),
        np.mean([len(t) for t in texts]), np.max([len(t) for t in texts]),
        sorted_scores[0], sorted_scores[0] - sorted_scores[min(1, len(scores)-1)],
        np.mean(sorted_scores[:5]), np.mean(sorted_scores[:10]), np.mean(scores == 0), entropy,
        coverage, len(a & b) / max(len(a | b), 1), len({paras[i]['section'] for i in rank[:5]})]
    patterns = (r'^(is|are|was|were|does|do|did|can|could|has|have)\b', r'\bhow many\b',
        r'\bhow (much|long|large|big)\b', r'\b(data|dataset|corpus|corpora)\b',
        r'\b(baseline|compare|comparison)\b', r'\b(metric|evaluation|accuracy)\b',
        r'\b(model|architecture|mechanism)\b', r'\b(language|languages)\b')
    values += [int(bool(re.search(p, q.lower()))) for p in patterns]
    values += list(np.log1p(lengths))
    return np.asarray(values, dtype=float), lengths, (time.perf_counter()-start)*1000, bm25


def summarize(rows):
    return {'papers': len(rows), 'answer_f1': float(np.mean([r['f1'] for r in rows])),
        'normalized_em': float(np.mean([r['em'] for r in rows])),
        'mean_answerer_tflops': float(np.mean([r['estimated_flops_v2']/1e12 for r in rows])),
        'mean_answerer_ms': float(np.mean([1000*r['answerer_total_seconds'] for r in rows])),
        'p95_answerer_ms': float(np.quantile([1000*r['answerer_total_seconds'] for r in rows], .95)),
        'actions': dict(Counter(r['mode'] for r in rows))}


def main():
    out = ROOT / 'results/qasper_cheap_control_v1'
    out.mkdir(parents=True, exist_ok=False)
    raw = list(map(json.loads, (ROOT/'data/qasper_long_audit_v1/selected_96.jsonl').read_text().splitlines()))
    scored = list(map(json.loads, (ROOT/'results/qasper_answerer_v1/scored.jsonl').read_text().splitlines()))
    by = {r['paper_id']: {} for r in raw}
    for row in scored:
        by[row['paper_id']][row['mode']] = row
    tok = AutoTokenizer.from_pretrained(ROOT/'data/raw/qwen4b_tokenizer', local_files_only=True)
    feature_rows = {}
    for row in raw:
        values, lengths, elapsed, bm25 = features(observation(row), tok)
        assert bm25 == row['bm25_ranked_paragraph_ids']
        assert lengths == [by[row['paper_id']][m]['prompt_tokens'] for m in MODES]
        feature_rows[row['paper_id']] = {'values': values.tolist(), 'prompt_tokens': lengths, 'cpu_ms': elapsed}
    cost = DenseForwardCost(**json.loads((ROOT/'captures/qasper_answerer_4b_v1/receipt.json').read_text())['cost_model'])
    result = {'scope': 'Exploratory out-of-fold CPU learned-Q; no trained LM or full-system measurement',
        'modes': MODES, 'betas': BETAS, 'features': len(next(iter(feature_rows.values()))['values']),
        'feature_mean_cpu_ms': float(np.mean([v['cpu_ms'] for v in feature_rows.values()])),
        'feature_p95_cpu_ms': float(np.quantile([v['cpu_ms'] for v in feature_rows.values()], .95)), 'slices': {}}
    saved = []
    for slice_name in ('original_96', 'reviewed_77'):
        ids = sorted([k for k in by if slice_name == 'original_96' or by[k]['full']['reviewed_primary']],
                     key=lambda k: hashlib.sha256(('cheap-control-v1:'+k).encode()).hexdigest())
        X = np.array([feature_rows[k]['values'] for k in ids])
        Y = np.array([[by[k][m]['f1'] for m in MODES] for k in ids])
        C = np.array([[by[k][m]['estimated_flops_v2']/1e12 for m in MODES] for k in ids])
        folds = np.arange(len(ids)) % 5
        selections = {f'{name}_beta{beta:g}': np.zeros(len(ids), dtype=int)
                      for name in ('ridge', 'trees', 'training_constant') for beta in BETAS}
        fit_times, inference_times = [], []
        for fold in range(5):
            tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            train_ids = [ids[i] for i in tr]; test_ids = [ids[i] for i in te]
            assert not set(train_ids) & set(test_ids)
            output_lengths = [int(round(np.mean([by[k][m]['generated_tokens'] for k in train_ids]))) for m in MODES]
            predicted_cost = np.array([[cost.generation(feature_rows[k]['prompt_tokens'][j], max(output_lengths[j], 1))/1e12
                                        for j in range(len(MODES))] for k in test_ids])
            for name, model in [('ridge', make_pipeline(StandardScaler(), Ridge(alpha=10))),
                    ('trees', ExtraTreesRegressor(n_estimators=128, max_depth=3, min_samples_leaf=5, random_state=1307, n_jobs=1))]:
                tick = time.perf_counter()
                with threadpool_limits(limits=1): model.fit(X[tr], Y[tr])
                fit_times.append({'fold': fold, 'model': name, 'seconds': time.perf_counter()-tick})
                tick = time.perf_counter()
                # Batch-one policy inference, so batching is not presented as request latency.
                with threadpool_limits(limits=1):
                    pred = np.clip(np.vstack([model.predict(X[i:i+1])[0] for i in te]), 0, 1)
                inference_times.append({'fold': fold, 'model': name, 'ms_per_paper': (time.perf_counter()-tick)*1000/len(te)})
                for beta in BETAS:
                    selections[f'{name}_beta{beta:g}'][te] = np.argmax(pred - beta*predicted_cost, axis=1)
                for i, qvals, predicted_work in zip(te, pred, predicted_cost):
                    saved.append({'slice': slice_name, 'paper_id': ids[i], 'fold': fold, 'model': name,
                        'predicted_f1': qvals.tolist(), 'predicted_answerer_tflops': predicted_work.tolist(),
                        'training_papers': train_ids})
            for beta in BETAS:
                selections[f'training_constant_beta{beta:g}'][te] = int(np.argmax(np.mean(Y[tr]-beta*C[tr], axis=0)))
        stats = {}
        rng = np.random.default_rng(1307)
        draws = rng.integers(0, len(ids), (10000, len(ids)))
        for name, actions in selections.items():
            chosen = [by[k][MODES[int(a)]] for k, a in zip(ids, actions)]
            s = summarize(chosen); s['comparisons'] = {}
            for base in ('full', 'bm25_10'):
                d = np.array([r['f1']-by[k][base]['f1'] for k,r in zip(ids,chosen)])
                work = np.array([r['estimated_flops_v2'] for r in chosen]); bwork = np.array([by[k][base]['estimated_flops_v2'] for k in ids])
                s['comparisons'][base] = {'f1_delta': float(d.mean()), 'f1_delta_95ci': np.quantile(d[draws].mean(axis=1), [.025,.975]).tolist(),
                    'work_ratio': float(work.mean()/bwork.mean()),
                    'work_ratio_95ci': np.quantile(work[draws].mean(axis=1)/bwork[draws].mean(axis=1), [.025,.975]).tolist()}
            stats[name] = s
        oracle = [max((by[k][m] for m in MODES), key=lambda r:(r['f1'], -r['estimated_flops_v2'])) for k in ids]
        result['slices'][slice_name] = {'folds': dict(zip(ids, folds.tolist())), 'policies': stats,
            'fixed_actions': {m:summarize([by[k][m] for k in ids]) for m in MODES},
            'nondeployable_hindsight_best': summarize(oracle), 'fitting': fit_times, 'batch_one_inference': inference_times}
    result['limits'] = ['Already inspected development data, not final validation.',
        'Oracle chooses using true per-action scores and pays zero selection cost; not an algorithm.',
        'All policy choices use held-out-paper predictions, but method design has seen aggregate development results.',
        'Intervals condition on trained folds; do not include training uncertainty or correct multiple comparisons.',
        'CPU features/model timing is local Mac timing, separate from archived GPU answerer latency.',
        'No new LM controller weights, online evidence service, or full-system latency measured.']
    (out/'features.json').write_text(json.dumps(feature_rows,indent=2)+'\n')
    (out/'oof_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in saved))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'feature_mean_cpu_ms':result['feature_mean_cpu_ms'],
        'reviewed_77_policies':{k:{a:v[a] for a in ('answer_f1','mean_answerer_tflops','actions')} for k,v in result['slices']['reviewed_77']['policies'].items()}},indent=2))


if __name__ == '__main__':
    main()
