"""Deterministic sample isolation. Never prints question, reference, or support text."""
import collections,hashlib,json,math,sqlite3,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'musique_pilot200_20261114'
SEED='evidence-control-data-v1-20261114'
def stable(s):return hashlib.sha256((SEED+'|'+s).encode()).hexdigest()
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def parts(r):return {p['id'] for p in r['question_decomposition']}
def titles(r):return {p['title'] for p in r['paragraphs'] if p['is_supporting']}
def doc(p):
    s=json.dumps([p['title'],p['paragraph_text']],ensure_ascii=False,separators=(',',':'))
    return dict(doc_id=hashlib.sha256(s.encode()).hexdigest(),title=p['title'],body=p['paragraph_text'],score=0.)
def write_rows(p,rows):
    with p.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def groups(rows):
    parent={r['id']:r['id'] for r in rows};owner={}
    def find(x):
        while x!=parent[x]:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for r in rows:
        for title in {('title',s) for s in titles(r)}|{('component',s) for s in parts(r)}:
            if title in owner:parent[find(r['id'])]=find(owner[title])
            else:owner[title]=r['id']
    return {q:find(q) for q in parent}
def select(rows,n,name,excluded,unique_components=True):
    blocked_ids={r['id'] for r in excluded}
    blocked_parts=set().union(*(parts(r) for r in excluded))
    blocked_titles=set().union(*(titles(r) for r in excluded))
    reasons=collections.Counter();eligible=[]
    for r in rows:
        why=[]
        if r['id'] in blocked_ids:why.append('question')
        if parts(r)&blocked_parts:why.append('component')
        if titles(r)&blocked_titles:why.append('source_title')
        if why:
            reasons.update(why);continue
        eligible.append(r)
    counts=collections.Counter(len(r['question_decomposition']) for r in eligible)
    exact={h:n*c/len(eligible) for h,c in counts.items()};quota={h:math.floor(v) for h,v in exact.items()}
    for h in sorted(exact,key=lambda h:(-(exact[h]-quota[h]),h))[:n-sum(quota.values())]:quota[h]+=1
    used=set();chosen=[];actual=collections.Counter();skipped=0
    for r in sorted(eligible,key=lambda r:((-len(r['question_decomposition']) if name=='fit' else 0),stable(name+'|'+r['id']))):
        h=len(r['question_decomposition'])
        if actual[h]>=quota[h]:continue
        if unique_components and parts(r)&used:skipped+=1;continue
        chosen.append(r);actual[h]+=1;used|=parts(r)
        if len(chosen)==n:break
    summary=dict(source_rows=len(rows),eligible_rows=len(eligible),exclusion_reason_counts=dict(reasons),
        eligible_hops=dict(counts),quota=dict(quota),actual=dict(actual),within_component_skips=skipped)
    assert len(chosen)==n and actual==quota,(name,summary)
    return chosen,summary

def main():
    dst=ROOT/'data/fresh_v3';dst.mkdir(parents=True,exist_ok=False)
    sealed=dst/'sealed';sealed.mkdir()
    prior=json.loads((OLD/'data_receipt.json').read_text())
    archive=OLD/'data/musique_v1.0.zip';assert digest(archive)==prior['archive_sha256']
    raw_train=OLD/'data/train.jsonl';assert digest(raw_train)==prior['train_sha256']
    train=[json.loads(s) for s in raw_train.read_text().splitlines()]
    with zipfile.ZipFile(archive) as z:
        dev_bytes=z.read('data/musique_ans_v1.0_dev.jsonl')
    dev=[json.loads(s) for s in dev_bytes.splitlines()]
    assert all(r['answerable'] for r in train+dev)
    assert len({r['id'] for r in train+dev})==len(train)+len(dev)
    old_ids=set(prior['pilot_ids']+prior['smoke_ids']);exposed=[r for r in train if r['id'] in old_ids]
    assert len(exposed)==202
    confirmation,cs=select(dev,600,'confirmation',exposed,unique_components=False)
    selection,vs=select(train,200,'selection',exposed+confirmation)
    fit,ts=select(train,800,'fit',exposed+confirmation+selection)
    splits=dict(fit=fit,selection=selection,confirmation=confirmation)
    fingerprints={};splitmeta={}
    for name,rs in splits.items():
        group=groups(rs);sizes=collections.Counter(group.values())
        def obs(r):
            docs=[doc(p) for p in r['paragraphs']]
            docs.sort(key=lambda d:stable(r['id']+'|native|'+d['doc_id']))
            return dict(id=r['id'],question=r['question'],native_documents=docs)
        op=dst/(name+'_observations.jsonl');write_rows(op,map(obs,rs))
        ep=sealed/(name+'_evaluator.jsonl')
        write_rows(ep,(dict(id=r['id'],question=r['question'],answer=r['answer'],answer_aliases=r['answer_aliases'],
            hops=len(r['question_decomposition']),support_ids=[doc(p)['doc_id'] for p in r['paragraphs'] if p['is_supporting']],
            source_group=group[r['id']],raw_record=r) for r in rs))
        review=sorted([r['id'] for r in rs],key=lambda q:stable('review|'+name+'|'+q))[:30 if name=='confirmation' else 20]
        splitmeta[name]=dict(ids=[r['id'] for r in rs],source_components=len(sizes),component_sizes=sorted(sizes.values(),reverse=True),review_ids=review)
        fingerprints[str(op.relative_to(dst))]=digest(op);fingerprints[str(ep.relative_to(dst))]=digest(ep)
    # Independent assertions across all selected splits, including the exposed set.
    names=list(splits)+['exposed'];checksets={**splits,'exposed':exposed}
    checks=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            overlap=dict(questions=len({r['id'] for r in checksets[a]}&{r['id'] for r in checksets[b]}),
                components=len(set().union(*(parts(r) for r in checksets[a]))&set().union(*(parts(r) for r in checksets[b]))),
                support_titles=len(set().union(*(titles(r) for r in checksets[a]))&set().union(*(titles(r) for r in checksets[b]))))
            assert not any(overlap.values()),(a,b,overlap)
            checks.append(dict(pair=[a,b],**overlap))
    documents={}
    for r in train+dev:
        for p in r['paragraphs']:
            d=doc(p);documents[d['doc_id']]=d
    con=sqlite3.connect(dst/'index.sqlite')
    con.execute('CREATE VIRTUAL TABLE documents USING fts5(doc_id UNINDEXED,title,body)')
    con.executemany('INSERT INTO documents(doc_id,title,body) VALUES (?,?,?)',
        [(d['doc_id'],d['title'],d['body']) for _,d in sorted(documents.items())])
    con.commit();con.execute("INSERT INTO documents(documents) VALUES('optimize')");con.commit();con.close()
    fingerprints['index.sqlite']=digest(dst/'index.sqlite')
    receipt=dict(version='v3',seed=SEED,archive_sha256=digest(archive),train_sha256=digest(raw_train),
        dev_member='data/musique_ans_v1.0_dev.jsonl',dev_member_sha256=hashlib.sha256(dev_bytes).hexdigest(),
        train_count=len(train),dev_count=len(dev),corpus_documents=len(documents),fingerprints=fingerprints,
        selection_summaries=dict(fit=ts,selection=vs,confirmation=cs),splits=splitmeta,cross_split_overlap=checks,
        confirmation_content_printed=False,confirmation_labels_scored=False,official_test_extracted=False)
    (ROOT/'experiments/fresh_data_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['train_count','dev_count','corpus_documents','selection_summaries','cross_split_overlap']},indent=2))
    print(json.dumps({k:dict(source_components=v['source_components'],largest_component=v['component_sizes'][0]) for k,v in splitmeta.items()}))
if __name__=='__main__':main()
