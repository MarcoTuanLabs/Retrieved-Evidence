"""Metadata-only freshness inventory; no model answers, metrics or new inference."""
import collections
import hashlib
import json
import time
import unicodedata
import zipfile
from pathlib import Path
from prepare_fresh_data import digest,titles

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/h25_natural_evidence_transfer'
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).replace('_',' ').casefold().split())
def support(r):
    facts=r['supporting_facts']
    ts=facts['title'] if isinstance(facts,dict) else [t for t,_ in facts]
    return {norm(t) for t in ts}
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    started=time.time();assert not (RUN/'DATA_INVENTORY.json').exists()
    base=ROOT/'experiments/h18_fresh_repair_confirmation/MUSIQUE_AVAILABILITY.json'
    inv=json.loads(base.read_text());blocked={norm(t) for t in inv['blocked_titles']};questions=set();used_two=set();sources=[base]
    labelpaths=sorted((ROOT/'data').glob('**/sealed/*evaluator.jsonl'))
    assert not any('h25' in str(p) for p in labelpaths)
    for p in labelpaths:
        sources.append(p)
        for row in read(p):
            raw=row.get('raw_record',row);questions.add(norm(row['question']))
            if 'paragraphs' in raw:blocked|={norm(t) for t in titles(raw)}
            elif 'supporting_facts' in raw:blocked|=support(raw)
            if 'twowiki' in p.parts:used_two.add(row['id'])
    old=ROOT.parent/'musique_pilot200_20261114';receipt=json.loads((old/'data_receipt.json').read_text());used=set(receipt['pilot_ids']+receipt['smoke_ids'])
    for row in read(old/'data/train.jsonl'):
        if row['id'] in used:questions.add(norm(row['question']));blocked|={norm(t) for t in titles(row)}
    sources += [old/'data_receipt.json',old/'data/train.jsonl']
    archive=ROOT/'data/twowiki_source/data_ids_april7.zip'
    expected=json.loads((ROOT/'experiments/h18_fresh_repair_confirmation/TWOWIKI_AVAILABILITY.json').read_text())['archive_sha256']
    assert digest(archive)==expected
    with zipfile.ZipFile(archive) as z:dev=json.loads(z.read('dev.json'))
    eligible=[];reasons=collections.Counter()
    for r in dev:
        excluded=[]
        if r['_id'] in used_two:excluded.append('previous_id')
        if norm(r['question']) in questions:excluded.append('previous_exact_question')
        if support(r)&blocked:excluded.append('previous_evaluated_support_title')
        reasons.update(excluded)
        if not excluded:eligible.append(r)
    selected=[];seen_titles=set();seen_questions=set();counts=collections.Counter()
    for r in sorted(eligible,key=lambda r:hashlib.sha256(('h25-transfer-1325|'+r['_id']).encode()).hexdigest()):
        if counts[r['type']]>=100 or support(r)&seen_titles or norm(r['question']) in seen_questions:continue
        selected.append(r);seen_titles|=support(r);seen_questions.add(norm(r['question']));counts[r['type']]+=1
    types=sorted({r['type'] for r in dev});assert len(types)==4
    feasible=len(selected)==400 and all(counts[t]==100 for t in types)
    result=dict(status='feasible' if feasible else 'quota_unavailable',created_unix=time.time(),
        all_dev=len(dev),prior_evaluated_2wiki_ids=len(used_two),blocked_support_titles=len(blocked),prior_exact_questions=len(questions),
        overlapping_exclusion_reason_counts=dict(reasons),eligible=len(eligible),eligible_types=dict(collections.Counter(r['type'] for r in eligible)),
        selected_count=len(selected),selected_types=dict(counts),selected_support_document_counts=dict(collections.Counter(len(support(r)) for r in selected)),
        selected_ids=[r['_id'] for r in selected],within_panel_support_title_overlap=0,within_panel_exact_question_overlap=0,
        selected_prior_support_title_overlap=sum(bool(support(r)&blocked) for r in selected),
        selected_prior_exact_question_overlap=sum(norm(r['question']) in questions for r in selected),
        excluded_title_scope='Supporting titles of previously evaluated questions; not exhaustive previous distractors or pretrained exposure',
        archive_sha256=expected,source_sha256={str(p):digest(p) for p in sources},inventory_source_sha256=digest(Path(__file__)),
        contract_sha256=digest(RUN/'CONTRACT.txt'),model_outcomes_read=False,new_model_calls=0,official_test_used=False,
        elapsed_seconds=time.time()-started)
    assert result['elapsed_seconds']<900
    save(RUN/'DATA_INVENTORY.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['selected_ids','source_sha256']},ensure_ascii=False))
    assert feasible,'Preserve inventory; do not relax exclusions or change quotas silently'

if __name__=='__main__':main()
