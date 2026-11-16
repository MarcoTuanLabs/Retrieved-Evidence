"""Freeze two cheap question-only rules before observing next-source outcomes."""
import hashlib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YESNO_PATTERN = r'^(is|are|was|were|does|do|did|can|could|has|have)\b'


def choose(question):
    # Exactly the already-used observable feature; no answer_type label or paper.
    flag = bool(re.search(YESNO_PATTERN, question.lower()))
    return {'question_rule_full': 'bm25_3' if flag else 'full',
            'question_rule_top10': 'bm25_3' if flag else 'bm25_10'}


def main():
    output = ROOT/'data/qasper_next96_question_rules_v1'
    output.mkdir(exist_ok=False)
    source = ROOT/'data/qasper_long_next96_v1/selected_96.jsonl'
    records = []
    for row in map(json.loads, source.read_text().splitlines()):
        start = time.perf_counter()
        actions = choose(row['question'])
        records.append({'paper_id':row['paper_id'], 'choices':actions,
                        'rule_cpu_ms':1000*(time.perf_counter()-start)})
    path = output/'choices.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    freeze = {'frozen_unix':time.time(), 'new_outcomes_inspected':False,
        'basis':'Old-source type decomposition only; added after capture launched while model was loading.',
        'pattern':YESNO_PATTERN, 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'choices_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'rules':{'question_rule_full':'observable yes/no => BM25-3; otherwise full',
                 'question_rule_top10':'observable yes/no => BM25-3; otherwise BM25-10'},
        'cost_limit':'Only rule timing measured here. Deployment charges chosen retrieval and answerer; never requires all30 learned features.',
        'scope':'Additional fixed development baselines, not new learned candidates; no threshold search.'}
    (output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    print(json.dumps(freeze))


if __name__=='__main__':main()
