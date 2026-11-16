"""Validate a complete, ordered checkpoint prefix without opening gold labels."""
import hashlib
import json
import math
from pathlib import Path

def validate_prefix(path, mirror):
    path=Path(path);mirror=Path(mirror)
    receipt=json.loads((path/'receipt.json').read_text())
    assert receipt['state']=='failed' and 'User wallclock deadline' in receipt.get('error','')
    backup=json.loads((path/'BACKUP.json').read_text());assert backup['verified']
    for name,h in backup['files'].items():
        for base in [path,mirror]:
            with (base/name).open('rb') as f:assert hashlib.sha256(f.read()).hexdigest()==h,name
    records=[json.loads(line) for line in (path/'records.jsonl').read_text().splitlines()]
    schedule=json.loads((path/'schedule.json').read_text())
    expected=[(e['dataset'],e['id'],m) for e in schedule for m in e['methods']]
    keys=[(r['dataset'],r['id'],r['method']) for r in records]
    assert 0<len(keys)<len(expected)==3200
    assert keys==expected[:len(keys)] and len(keys)==len(set(keys))
    assert receipt['completed']==len(records) and receipt['completed_questions']==len(records)//4
    counts=dict(completed_writers=sum(r['writer'] is not None for r in records),
        completed_readers=sum(not r['accepted_by_rule'] for r in records),
        completed_rule_answers=sum(r['accepted_by_rule'] for r in records),
        completed_rankings=sum(r['legacy_prefix'] is not None and r['legacy_prefix']['ranking'] is not None for r in records),
        physical_sparse_queries=sum(len(r['physical_acquisitions']) for r in records))
    for k,v in counts.items():assert receipt[k]==v,k
    assert math.isclose(receipt['campaign_main_model_flops'],sum(r['total_model_flops'] for r in records),rel_tol=1e-12)
    return receipt,set(keys),schedule
