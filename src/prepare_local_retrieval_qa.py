"""Length/identity preparation from completed captures; no answers, new queries or GPU."""
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = ('Answer the question using the supplied passages when available. '
          'Give your best answer as a concise phrase or short list; for a yes/no question answer only yes or no. '
          'Output only the answer, with no explanation, preamble, or invented citations.')
POLICIES = ('question_only', 'bm25_5', 'bm25_20', 'paged_4x5', 'feedback_4x5')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def messages_for(question, documents):
    evidence = '\n\n'.join('Title: '+d['title']+'\n'+d['body'] for d in documents)
    user = ('Passages:\n'+evidence+'\n\n' if documents else '')+'Question: '+question
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]


def main():
    directory = ROOT/'results/local_retrieval_v1'
    observations_path = ROOT/'data/local_retrieval_v1/observations.jsonl'
    observations = {r['id']: r for r in map(json.loads, observations_path.read_text().splitlines())}
    assert len(observations) == len({r['group'] for r in observations.values()}) == 64
    assert all(set(r) == {'id', 'group', 'question'} for r in observations.values())
    trajectories = [r for r in map(json.loads, (directory/'trajectories.jsonl').read_text().splitlines()) if r['repetition'] == 0]
    paths, checked = {}, set()
    for row in trajectories:
        docs = {}
        for call in row['calls']:
            key = call['response_sha256']
            body = (directory/'responses'/(key+'.json')).read_bytes().rstrip(b'\n')
            assert hashlib.sha256(body).hexdigest() == key
            content = json.loads(body)
            assert content['request'] == call['request']
            checked.add(key)
            for document in content['documents']:
                assert set(document) == {'doc_id', 'title', 'body', 'score'}
                docs[document['doc_id']] = document
        paths[row['id'], row['policy']] = list(docs.values())
    tok = AutoTokenizer.from_pretrained(ROOT/'data/raw/qwen4b_tokenizer', local_files_only=True)
    records = []
    for key, observation in sorted(observations.items()):
        for policy in POLICIES:
            documents = [] if policy == 'question_only' else paths[key, policy]
            messages = messages_for(observation['question'], documents)
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            records.append({'id': key, 'group': observation['group'], 'policy': policy,
                'request_id': hashlib.sha256(canonical(messages).encode()).hexdigest(),
                'prompt_tokens': len(ids), 'documents': len(documents),
                'ordered_document_ids': [d['doc_id'] for d in documents]})
    by = {(r['id'], r['policy']): r for r in records}
    assert len(records) == 320
    assert all(by[q, 'bm25_20']['request_id'] == by[q, 'paged_4x5']['request_id'] for q in observations)
    result = {'scope': 'Local input preparation only; no model inference, gold labels, new service query or held-out sources.',
        'sources': 64, 'prompt_conditions': len(records), 'unique_prompts': len({r['request_id'] for r in records}),
        'response_hashes_checked': len(checked), 'paged_and_one_query_prompts_identical': True,
        'observations_sha256': hashlib.sha256(observations_path.read_bytes()).hexdigest(),
        'system_prompt': SYSTEM, 'policies': {}}
    for policy in POLICIES:
        lengths = np.array([r['prompt_tokens'] for r in records if r['policy'] == policy])
        result['policies'][policy] = {'mean_prompt_tokens': float(lengths.mean()),
            'minimum': int(lengths.min()), 'maximum': int(lengths.max()),
            'quantiles_25_50_75_95': np.quantile(lengths, [.25, .5, .75, .95]).tolist()}
    out = ROOT/'data/local_retrieval_qa_preflight_v1'
    out.mkdir(exist_ok=False)
    (out/'prompt_contract.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (out/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
