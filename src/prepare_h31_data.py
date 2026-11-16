"""Seal only the already-selected independent TwoWiki confirmation labels."""
import collections
import json
import os
import sqlite3
import time
import zipfile
from pathlib import Path

from prepare_fresh_data import digest,doc
from inventory_h25_data import support,norm

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/h31_fixed_policy_confirmation'
DATA=ROOT/'data/h31_confirmation'


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    start=time.time();assert not DATA.exists()
    inv=json.loads((RUN/'DATA_INVENTORY.json').read_text());assert inv['status']=='feasible'
    assert digest(RUN/'DATA_CONTRACT.json')==inv['data_contract_sha256']
    assert digest(ROOT/'src/inventory_h31_confirmation.py')==inv['inventory_source_sha256']
    for p,h in inv['source_sha256'].items():assert digest(Path(p))==h,p
    archive=ROOT/'data/twowiki_source/data_ids_april7.zip';assert digest(archive)==inv['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        dev={r['_id']:r for r in json.loads(z.read('dev.json'))};aliases={}
        for line in z.read('id_aliases.json').splitlines():
            r=json.loads(line);aliases[r['Q_id']]=set(r['aliases']+r['demonyms'])
    chosen=[dev[q] for q in inv['twowiki']['selected_ids']];assert len(chosen)==800
    assert dict(collections.Counter(r['type'] for r in chosen))==inv['twowiki']['selected_types']
    seen=set();qs=set();labels=[];observations=[]
    for row in chosen:
        st=support(row);assert not st&seen;seen|=st
        q=norm(row['question']);assert q not in qs;qs.add(q)
        raw_titles={t for t,_ in row['supporting_facts']}
        supports=[doc(dict(title=t,paragraph_text=''.join(ss)))['doc_id'] for t,ss in row['context'] if t in raw_titles]
        assert len(set(supports))==len(raw_titles)
        labels.append(dict(id=row['_id'],dataset='twowiki',question=row['question'],answer=row['answer'],answer_aliases=sorted(aliases.get(row['answer_id'],set())-{row['answer']}),
            source_group=row['_id'],type=row['type'],hops=len(raw_titles),hops_scope='Official supporting-document count, not guaranteed reasoning-hop count',support_ids=supports,raw_record=row))
        observations.append(dict(id=row['_id'],dataset='twowiki',question=row['question']))
    DATA.mkdir();(DATA/'sealed').mkdir()
    for name,rows in [('observations.jsonl',observations),('sealed/evaluator.jsonl',labels)]:
        (DATA/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(rows,key=lambda r:r['id'])))
    source=ROOT/'experiments/h27_strong_direct_retrieval/payload/indices/twowiki.sqlite'
    spec=json.loads((ROOT/'experiments/h27_strong_direct_retrieval/FROZEN.json').read_text())['configuration']['datasets']['twowiki']
    assert digest(source)==spec['index_sha256'];os.link(source,DATA/'index.sqlite')
    db=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True);available={r[0] for r in db.execute('SELECT doc_id FROM documents')};db.close()
    required={d for r in labels for d in r['support_ids']};assert required<=available
    result=dict(status='prepared',created_unix=time.time(),questions=800,datasets=dict(twowiki=dict(questions=800,documents=len(available),types=dict(collections.Counter(r['type'] for r in labels)),source_groups=800,
        annotated_support_documents=len(required),annotated_supports_in_corpus=True,index_sha256=spec['index_sha256'])),
        prior_question_overlap=0,prior_evaluated_support_title_overlap=0,within_panel_support_title_overlap=0,
        source_scope=inv['twowiki']['source_scope'],musique_source_disjoint_available=0,musique_independent_confirmation=False,
        fingerprints={str(p.relative_to(DATA)):digest(p) for p in DATA.rglob('*') if p.is_file()},
        inventory_sha256=digest(RUN/'DATA_INVENTORY.json'),preparation_source_sha256=digest(Path(__file__)),
        new_model_calls=0,model_outcomes_scored=False,unused_answers_printed=False,official_test_used=False,elapsed_seconds=time.time()-start)
    save(RUN/'DATA.json',result)
    with (RUN/'ledger.jsonl').open('a') as f:f.write(json.dumps(dict(event='confirmation_data_sealed',created_unix=time.time(),questions=800,source_groups=800,new_model_calls=0,status='prepared'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fingerprints'}))


if __name__=='__main__':main()
