"""Freeze natural long-document answerer comparisons before GPU inference.

All deployable selectors use only the question/document. Oracle evidence is
explicitly diagnostic; it is never presented as a deployed controller.
"""
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = ('Answer the question about the research paper. Use the supplied paper text when available. '
          'Give your best answer as a concise phrase or short list; for a yes/no question answer only yes or no. '
          'Output only the answer, with no explanation, preamble, or invented citations.')


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def section_preferences(question):
    q = question.lower()
    if re.search(r'\b(data|datasets?|corpus|corpora|collect|annotat|how big|how many)\b', q):
        return ('data', 'corpus', 'corpora', 'experiment', 'annotat')
    if re.search(r'metric|evaluat|baseline|compare', q):
        return ('evaluat', 'experiment', 'result', 'baseline')
    if re.search(r'better|improv|outperform|accuracy|performance', q):
        return ('result', 'experiment', 'discussion', 'conclusion')
    if re.search(r'architect|model|mechanism|objective|loss|train|feature', q):
        return ('method', 'model', 'approach', 'framework', 'train')
    return ('introduction', 'conclusion')


def main():
    data = ROOT / 'data/qasper_long_audit_v1'
    out = ROOT / 'data/qasper_answerer_pilot_v1'
    out.mkdir(parents=True, exist_ok=False)
    rows = list(map(json.loads, (data / 'selected_96.jsonl').read_text().splitlines()))
    review = {r['question_id']: r for r in map(json.loads, (data / 'semantic_review.jsonl').read_text().splitlines())}
    tok = AutoTokenizer.from_pretrained(ROOT / 'data/raw/qwen4b_tokenizer', local_files_only=True)
    requests, scoring, selections = {}, [], []
    rng = random.Random(1307)
    rng.shuffle(rows)
    for row in rows:
        paras = row['paragraphs']
        texts = [p['text'] for p in paras]
        vectorizer = TfidfVectorizer(decode_error='replace', strip_accents='unicode', analyzer='word', stop_words='english')
        index = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(index, vectorizer.transform([row['question']])).flatten()
        lexical = sorted(range(len(paras)), key=lambda i: (-similarity[i], paras[i]['id']))
        preferred = section_preferences(row['question'])
        section_scores = [float(similarity[i]) + .15 * any(s in paras[i]['section'].lower() for s in preferred) for i in range(len(paras))]
        section_order = sorted(range(len(paras)), key=lambda i: (-section_scores[i], paras[i]['id']))
        choices = {
            'question_only': [], 'abstract': ['abstract'], 'full': [p['id'] for p in paras],
            'bm25_3': row['bm25_ranked_paragraph_ids'][:3],
            'bm25_5': row['bm25_ranked_paragraph_ids'][:5],
            'bm25_10': row['bm25_ranked_paragraph_ids'][:10],
            'tfidf_5': [paras[i]['id'] for i in lexical[:5]],
            'section_5': [paras[i]['id'] for i in section_order[:5]],
            'oracle_evidence': row['evidence_ids'],
        }
        modes = list(choices)
        rng.shuffle(modes)
        for mode in modes:
            chosen = set(choices[mode])
            evidence = '\n\n'.join(p['section'] + '\n' + p['text'] for p in paras if p['id'] in chosen)
            user = 'Question: ' + row['question']
            if mode != 'question_only':
                user = 'Paper: ' + row['title'] + '\n\nPaper text:\n' + evidence + '\n\n' + user
            messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
            request_id = hashlib.sha256(canonical(messages).encode()).hexdigest()
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            requests.setdefault(request_id, {'request_id': request_id, 'messages': messages, 'expected_prompt_tokens': len(ids)})
            scoring.append({'request_id': request_id, 'paper_id': row['paper_id'], 'question_id': row['question_id'],
                'mode': mode, 'references': row['references'], 'answer_type': row['answer_type'],
                'reviewed_primary': review[row['question_id']]['status'] == 'include_in_reviewed_primary',
                'selected_paragraph_ids': choices[mode], 'gold_evidence_ids': row['evidence_ids']})
            selections.append({'paper_id': row['paper_id'], 'mode': mode, 'prompt_tokens': len(ids),
                'selected_paragraphs': len(chosen), 'all_annotated_evidence': set(row['evidence_ids']) <= chosen,
                'uses_gold_to_select_evidence': mode == 'oracle_evidence'})
    for name, values in [('requests.jsonl', list(requests.values())), ('scoring_index.jsonl', scoring), ('selection_diagnostics.jsonl', selections)]:
        (out / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in values))
    assert len(scoring) == 96 * 9 and sum(r['reviewed_primary'] for r in scoring) == 77 * 9
    report = {'created_unix': time.time(), 'source_papers': 96, 'reviewed_primary_papers': 77,
        'unique_requests': len(requests), 'scoring_rows': len(scoring), 'natural_answers_no_options': True,
        'model': 'Qwen/Qwen3-4B-Instruct-2507', 'revision': 'cdbee75f17c01a7cc42f958dc650907174af0554',
        'modes': sorted({r['mode'] for r in scoring}), 'max_new_tokens': 64,
        'sampling': 'greedy', 'precision': 'bfloat16', 'batch_size': 1, 'cross_query_kv_cache': False,
        'purpose': 'development answerer/reading-cost feasibility, NOT trained-controller or whole-system win',
        'model_outputs_seen': False, 'final_holdout_used': False,
        'prompt_token_stats': {mode: {'mean': float(np.mean([x['prompt_tokens'] for x in selections if x['mode'] == mode])),
            'max': max(x['prompt_tokens'] for x in selections if x['mode'] == mode)} for mode in sorted({x['mode'] for x in selections})},
        'hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.jsonl')},
        'review_sha256': hashlib.sha256((data / 'semantic_review.jsonl').read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limits': ['Original 96 and reviewed 77 are both reported; exclusion mask fixed before model outcomes.',
            'Oracle evidence uses annotations, is diagnostic, and is not deployable.',
            'TFIDF follows the authors lexical recipe but extends to top-5 and includes abstract; not an exact reproduction of their one-paragraph result.',
            'Cheap section prior is a fixed .15 score bonus based on observable question/header words, not a tuned strong controller.',
            'Answerer-only timings exclude real selection/retrieval/control and cannot establish end-to-end speedup.',
            'Original references and official scorer are unchanged; incomplete references and lexical F1 limits remain visible.']}
    (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
