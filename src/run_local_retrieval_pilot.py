"""Interleaved, real HTTP retrieval feasibility; no LLM quality/speed claims.

Gold is first loaded AFTER every retrieval trajectory has finished. Feedback
query proposals inspect only the question and already returned natural text.
"""
import hashlib
import http.client
import json
import platform
import random
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from local_retrieval_server import STOPWORDS

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('bm25_5', 'bm25_20', 'paged_4x5', 'feedback_4x5')


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def entity_feedback(question, documents, used):
    # Deliberately cheap deterministic comparator, not a learned entity extractor.
    # This can fail; do not tune it on the debug outcomes and call it confirmatory.
    text = '\n'.join(d['body'] for d in list(documents.values())[:5])
    candidates = re.findall(r'\b[A-Z][a-zA-Z]+(?:[ -]+(?:[A-Z][a-zA-Z]+|of|the|and)){1,4}', text)
    counts = Counter(s.strip() for s in candidates)
    viable = [s for s in counts if s.lower() not in used and s.lower() not in question.lower()
              and any(w.lower() not in STOPWORDS for w in s.split())]
    if not viable:
        return question
    chosen = min(viable, key=lambda s: (-counts[s], -len(s.split()), s))
    used.add(chosen.lower())
    return chosen


def main():
    data = ROOT / 'data/local_retrieval_v1'
    out = ROOT / 'results/local_retrieval_v1'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'responses').mkdir()
    manifest = json.loads((data / 'manifest.json').read_text())
    observations = list(map(json.loads, (data / 'observations.jsonl').read_text().splitlines()))
    started = time.perf_counter()
    process = subprocess.Popen([sys.executable, str(ROOT / 'src/local_retrieval_server.py'),
        '--index', str(data / 'index.sqlite'), '--log', str(out / 'server_requests.jsonl')],
        stdout=subprocess.PIPE, stderr=(out / 'server_stderr.txt').open('w'), text=True)
    client = None
    records = []
    attempts, failed_attempts = 0, 0
    try:
        ready = json.loads(process.stdout.readline())
        assert ready['index_sha256'] == manifest['files']['index.sqlite']
        client = http.client.HTTPConnection('127.0.0.1', ready['port'], timeout=10)
        client.connect()
        client.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        def call(query, k, excluded):
            nonlocal attempts, failed_attempts
            payload = {'query': query, 'k': k, 'exclude_doc_ids': excluded}
            timing, errors = [], []
            for attempt in range(2):
                tick = time.perf_counter()
                attempts += 1
                try:
                    client.request('POST', '/search', body=canonical(payload).encode(), headers={'Content-Type': 'application/json'})
                    response = client.getresponse()
                    body = response.read()
                    timing.append((time.perf_counter() - tick) * 1000)
                    if response.status != 200:
                        raise RuntimeError('HTTP ' + str(response.status))
                    result = json.loads(body)
                    assert result['index_sha256'] == ready['index_sha256']
                    content = {'request': payload, 'documents': result['documents'], 'fts_query': result['fts_query']}
                    return result['documents'], {'request': payload, '_capture': content,
                        'http_ms': sum(timing), 'query_cpu_ms': result['query_cpu_ms'],
                        'query_wall_ms': result['query_wall_ms'], 'bytes': len(body),
                        'attempts': attempt + 1, 'errors': errors, 'no_hit': not result['documents']}
                except (OSError, http.client.HTTPException, RuntimeError) as error:
                    failed_attempts += 1
                    errors.append(type(error).__name__ + ': ' + str(error))
                    if len(timing) < attempt + 1:
                        timing.append((time.perf_counter() - tick) * 1000)
                    client.close()
                    if attempt == 1:
                        return [], {'request': payload, 'response_sha256': None, 'http_ms': sum(timing),
                            'query_cpu_ms': 0, 'query_wall_ms': 0, 'bytes': 0, 'attempts': 2,
                            'errors': errors, 'no_hit': True, 'terminal_failure': True}
                    client.connect()
                    client.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            raise AssertionError('unreachable')
        # Pipeline warmup is documented; it uses no selected task or gold.
        call('encyclopedia', 1, [])
        startup_ms = (time.perf_counter() - started) * 1000
        rng = random.Random(1307)
        schedule = []
        for rep in range(5):
            questions = list(observations)
            rng.shuffle(questions)
            for obs in questions:
                policies = list(POLICIES)
                rng.shuffle(policies)
                schedule.extend((rep, obs, policy) for policy in policies)
        (out / 'schedule.json').write_text(canonical([(rep, obs['id'], policy) for rep, obs, policy in schedule]) + '\n')
        with (out / 'trajectories.jsonl').open('x') as stream:
            for rep, obs, policy in schedule:
                tick = time.perf_counter()
                documents, calls, used = {}, [], set()
                rounds = 4 if policy in ('paged_4x5', 'feedback_4x5') else 1
                for step in range(rounds):
                    query = obs['question']
                    if policy == 'feedback_4x5' and step:
                        query = entity_feedback(query, documents, used)
                    returned, receipt = call(query, 20 if policy == 'bm25_20' else 5, list(documents))
                    calls.append(receipt)
                    for d in returned:
                        documents[d['doc_id']] = d
                elapsed = (time.perf_counter() - tick) * 1000
                # Evidence-archive hashing and disk writes are experimental
                # instrumentation and occur after the task timer stops.
                for receipt in calls:
                    content = receipt.pop('_capture', None)
                    if content is not None:
                        serialized = canonical(content)
                        key = hashlib.sha256(serialized.encode()).hexdigest()
                        receipt['response_sha256'] = key
                        path = out / 'responses' / (key + '.json')
                        if not path.exists():
                            path.write_text(serialized + '\n')
                record = {'id': obs['id'], 'group': obs['group'], 'repetition': rep, 'policy': policy,
                    'wall_ms': elapsed,
                    'titles': sorted({d['title'] for d in documents.values()}), 'documents': len(documents),
                    'evidence_chars': sum(len(d['title']) + len(d['body']) for d in documents.values()),
                    'calls': calls, 'created_unix': time.time()}
                stream.write(canonical(record) + '\n')
                records.append(record)
    finally:
        if client:
            client.close()
        process.terminate()
        process.wait(timeout=5)
    # Evaluator-only ground truth is loaded here, after all policies have run.
    gold = {x['id']: x for x in map(json.loads, (data / 'evaluator_only.jsonl').read_text().splitlines())}
    for record in records:
        supports = set(gold[record['id']]['support_titles'])
        hits = len(supports & set(record['titles']))
        record['support_recall'] = hits / len(supports)
        record['all_supports_found'] = hits == len(supports)
    summary = {'kind': 'Actual local HTTP FTS5/BM25 retrieval feasibility, not downstream QA',
        'corpus': manifest, 'host': platform.platform(), 'python': sys.version,
        'timing_repetitions': 5, 'independent_source_groups': len({r['group'] for r in records}),
        'trajectories': len(records), 'startup_warmup_ms': startup_ms, 'http_attempts_including_warmup': attempts,
        'failed_attempts': failed_attempts, 'artificial_delay': False, 'gpu_used': False,
        'raw_service_latency_is_not_whole_LLM_task_latency': True, 'policies': {}}
    for policy in POLICIES:
        rows = [r for r in records if r['policy'] == policy]
        times = [r['wall_ms'] for r in rows]
        summary['policies'][policy] = {'all_supports_found': float(np.mean([r['all_supports_found'] for r in rows])),
            'mean_support_recall': float(np.mean([r['support_recall'] for r in rows])),
            'mean_ms': float(np.mean(times)), 'p95_ms': float(np.quantile(times, .95)),
            'mean_evidence_chars': float(np.mean([r['evidence_chars'] for r in rows])),
            'mean_documents': float(np.mean([r['documents'] for r in rows])),
            'mean_server_cpu_ms': float(np.mean([sum(c['query_cpu_ms'] for c in r['calls']) for r in rows])),
            'no_hit_queries': sum(c['no_hit'] for r in rows for c in r['calls']),
            'retry_attempts': sum(c['attempts'] - 1 for r in rows for c in r['calls'])}
    # Repetitions must reproduce document content; timing never creates more n.
    parity = {}
    for policy in POLICIES:
        parity[policy] = all(len({tuple(r['titles']) for r in records if r['policy'] == policy and r['id'] == q}) == 1 for q in gold)
    summary['repeated_query_content_parity'] = parity
    assert all(parity.values()) and len(records) == 64 * 4 * 5
    # Paired source-group intervals for retrieval outcomes; no generalization claim.
    keys = sorted(gold)
    assert len({gold[k]['group'] for k in keys}) == len(keys)
    rng = np.random.default_rng(1307)
    indices = rng.integers(0, len(keys), size=(10000, len(keys)))
    base = {r['id']: r for r in records if r['policy'] == 'bm25_20' and r['repetition'] == 0}
    summary['paired_vs_bm25_20'] = {}
    for policy in POLICIES:
        rows = {r['id']: r for r in records if r['policy'] == policy and r['repetition'] == 0}
        differences = np.array([int(rows[k]['all_supports_found']) - int(base[k]['all_supports_found']) for k in keys])
        ci = np.quantile(differences[indices].mean(axis=1), [.025, .975]).tolist()
        summary['paired_vs_bm25_20'][policy] = {'all_supports_delta': float(differences.mean()), 'source_paired_95ci': ci}
    summary['limits'] = ['64 already-open source-title groups, not a final test.',
        'Support-title recall is a proxy, not QA accuracy or evidence sufficiency.',
        'Restricted curated-context corpus, not full Wikipedia.',
        'Simple feedback heuristic is not a trained controller or optimized strongest baseline.',
        'No external network service, GPU inference, training, or full task speed result.',
        'Natural event independence remains unaudited beyond source-title grouping.']
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'corpus'}, indent=2))


if __name__ == '__main__':
    main()
