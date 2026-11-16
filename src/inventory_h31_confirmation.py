"""Metadata-only inventory and fixed selection; never print unused answers/text."""
import collections
import hashlib
import json
import time
import zipfile
from pathlib import Path

from prepare_fresh_data import digest,titles,parts
from inventory_h25_data import norm,support

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/h31_fixed_policy_confirmation'
OLD=ROOT.parent/'musique_pilot200_20261114'


def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    start=time.time();contract=json.loads((RUN/'DATA_CONTRACT.json').read_text())
    assert not (RUN/'DATA_INVENTORY.json').exists()
    base=ROOT/'experiments/h18_fresh_repair_confirmation/MUSIQUE_AVAILABILITY.json'
    prior=json.loads(base.read_text());blocked={norm(t) for t in prior['blocked_titles']};components=set(prior['blocked_components'])
    questions=set();used_two=set();used_mu=set();sources=[base]
    labelpaths=sorted((ROOT/'data').glob('**/sealed/*evaluator.jsonl'))
    for p in labelpaths:
        assert 'h31' not in str(p);sources.append(p)
        for row in read(p):
            raw=row.get('raw_record',row);questions.add(norm(row['question']))
            if 'paragraphs' in raw:
                used_mu.add(row['id']);blocked|={norm(t) for t in titles(raw)};components|=parts(raw)
            elif 'supporting_facts' in raw:
                blocked|=support(raw)
                if 'twowiki' in p.parts:used_two.add(row['id'])
    receipt_path=OLD/'data_receipt.json';receipt=json.loads(receipt_path.read_text());old_ids=set(receipt['pilot_ids']+receipt['smoke_ids']);used_mu|=old_ids
    train_path=OLD/'data/train.jsonl';train=read(train_path)
    for row in train:
        if row['id'] in old_ids:
            questions.add(norm(row['question']));blocked|={norm(t) for t in titles(row)};components|=parts(row)
    sources += [receipt_path,train_path]
    mu_archive=OLD/'data/musique_v1.0.zip';assert digest(mu_archive)==receipt['archive_sha256'];sources.append(mu_archive)
    with zipfile.ZipFile(mu_archive) as z:mu_dev=[json.loads(l) for l in z.read('data/musique_ans_v1.0_dev.jsonl').splitlines()]
    mu_summary={}
    for split,rows in [('train',train),('dev',mu_dev)]:
        unused=[r for r in rows if r['id'] not in used_mu and norm(r['question']) not in questions]
        isolated=[r for r in unused if not {norm(t) for t in titles(r)}&blocked and not parts(r)&components]
        mu_summary[split]=dict(total=len(rows),unused_questions=len(unused),unused_hops=dict(collections.Counter(len(r['question_decomposition']) for r in unused)),
            unused_source_disjoint=len(isolated),source_disjoint_hops=dict(collections.Counter(len(r['question_decomposition']) for r in isolated)))
    archive=ROOT/'data/twowiki_source/data_ids_april7.zip'
    expected=json.loads((ROOT/'experiments/h18_fresh_repair_confirmation/TWOWIKI_AVAILABILITY.json').read_text())['archive_sha256']
    assert digest(archive)==expected
    with zipfile.ZipFile(archive) as z:dev=json.loads(z.read('dev.json'))
    candidates=[];reasons=collections.Counter()
    for row in dev:
        excluded=[]
        if row['_id'] in used_two:excluded.append('prior_id')
        if norm(row['question']) in questions:excluded.append('prior_exact_question')
        if support(row)&blocked:excluded.append('prior_evaluated_support_title')
        reasons.update(excluded)
        if not excluded:candidates.append(row)
    chosen=[];selected_titles=set();selected_questions=set();counts=collections.Counter()
    for row in sorted(candidates,key=lambda r:hashlib.sha256((contract['selection_seed']+'|'+r['_id']).encode()).hexdigest()):
        typ=row['type'];ts=support(row);question=norm(row['question'])
        if counts[typ]>=contract['twowiki_quota_per_type'] or ts&selected_titles or question in selected_questions:continue
        chosen.append(row);counts[typ]+=1;selected_titles|=ts;selected_questions.add(question)
    types=sorted({r['type'] for r in dev});assert len(types)==4
    feasible=len(chosen)==contract['twowiki_questions'] and all(counts[t]==contract['twowiki_quota_per_type'] for t in types)
    result=dict(status='feasible' if feasible else 'quota_unavailable',created_unix=time.time(),
        twowiki=dict(all_dev=len(dev),used_ids=len(used_two),eligible=len(candidates),eligible_types=dict(collections.Counter(r['type'] for r in candidates)),
            exclusions=dict(reasons),selected=len(chosen),selected_types=dict(counts),selected_ids=[r['_id'] for r in chosen],
            selected_prior_question_overlap=sum(norm(r['question']) in questions for r in chosen),
            selected_prior_support_title_overlap=sum(bool(support(r)&blocked) for r in chosen),within_selected_support_title_overlap=0,
            source_scope='Disjoint IDs, normalized exact questions and evaluated supporting titles; not an unseen-corpus, unseen-distractor or pretraining-independence guarantee.'),
        musique=mu_summary,blocked_prior_support_titles=len(blocked),blocked_components=len(components),prior_questions=len(questions),
        metadata_only=True,unused_answers_printed=False,new_model_calls=0,model_outcomes_read=False,official_test_used=False,
        source_sha256={str(p):digest(p) for p in sources},archive_sha256=expected,
        inventory_source_sha256=digest(Path(__file__)),data_contract_sha256=digest(RUN/'DATA_CONTRACT.json'),elapsed_seconds=time.time()-start)
    assert result['elapsed_seconds']<900
    save(RUN/'DATA_INVENTORY.json',result)
    with (RUN/'ledger.jsonl').open('a') as f:f.write(json.dumps(dict(event='data_inventory_complete',created_unix=time.time(),status=result['status'],selected=len(chosen),metadata_only=True,new_model_calls=0))+'\n')
    printable=dict(result);printable.pop('source_sha256');printable['twowiki']={k:v for k,v in result['twowiki'].items() if k!='selected_ids'}
    print(json.dumps(printable))
    assert feasible,'Do not relax exclusions or change quotas silently.'


if __name__=='__main__':main()
