"""One intervention: skip the unused 4B draft before fixed protected10 answering."""
from h18_pipeline import execute as paid_execute
from strategy_capabilities import bounded_messages, parse_writer

METHODS = ['paid_draft_fixed10', 'no_draft_fixed10']


def execute(observation, method, search, backend4, backend8):
    assert set(observation) == {'id', 'question', 'dataset'} and method in METHODS
    if method == 'paid_draft_fixed10':
        result = paid_execute(observation, 'fixed_upgrade8', search, backend4, backend8, None, None)
        result['draft_skipped'] = False
        return result
    question = observation['question']
    calls, acquisitions, events = [], [], []

    def acquire(query, k, excluded):
        docs, receipt = search(query, k, excluded)
        acquisitions.append(receipt)
        events.append(dict(event='search', number=len(acquisitions), query=query,
                           documents=[d['doc_id'] for d in docs]))
        return docs

    def call(backend, documents, role, history=None):
        messages, visibility = bounded_messages(question, documents, backend.tokenizer, history, False)
        value = backend.answer(messages).value
        value.update(role=role, visibility=visibility, _messages=messages)
        calls.append(value)
        events.append(dict(event=role, call=len(calls), request_id=value['request_id']))
        return value

    initial = acquire(question, 5, [])
    writer = call(backend4, initial, 'query_writer', [question])
    query, parse = parse_writer(writer['text'], question, False)
    new = acquire(query, min(15, 20-len(initial)), [d['doc_id'] for d in initial])
    assert not {d['doc_id'] for d in initial} & {d['doc_id'] for d in new}
    ranking, rank_cost, kept = None, 0, []
    if new:
        ranking = backend4.rank(dict(question=query, paragraphs=[
            dict(id=d['doc_id'], section=d['title'], text=d['body']) for d in new])).value
        order = ranking['ranked_paragraph_ids']
        assert set(order) == {d['doc_id'] for d in new}
        keep = set(order[:5])
        kept = [d for d in new if d['doc_id'] in keep]
        rank_cost = ranking['estimated_linear_attention_flops']
        events.append(dict(event='ranking', selected=[d['doc_id'] for d in kept]))
    protected, pool = initial+kept, initial+new
    events.append(dict(event='decision', action='upgrade8', model_calls_so_far=len(calls),
                       searches_so_far=len(acquisitions)))
    answer = call(backend8, protected, 'answer8')
    assert [c['role'] for c in calls] == ['query_writer', 'answer8']
    return dict(answer=answer['text'], action='upgrade8', calls=calls, acquisition_calls=acquisitions,
                events=events, initial_documents=initial, acquired_documents=pool,
                protected_documents=protected, final_documents=protected, ranking=ranking,
                writer_parse=parse, route=dict(action='upgrade8'), decision_observation=None,
                decision_seconds=0, prefix_model_flops=writer['model_flops']+rank_cost,
                draft_model_flops=0, ranker_model_flops=rank_cost, head_flops=0,
                total_model_flops=rank_cost+sum(c['model_flops'] for c in calls),
                selected_ids=answer['visibility']['visible_document_ids'], visibility=answer['visibility'],
                draft_skipped=True)


def self_check():
    """Check intervention, prompt identity and full-cost accounting on varied contexts."""
    import hashlib
    import json
    from types import SimpleNamespace

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2, 3]

    class Backend:
        tokenizer = Tokenizer()
        def __init__(self, big=False):
            self.big, self.n = big, 0
        def answer(self, messages):
            self.n += 1
            text = 'Synthetic answer' if self.big else ('{"query":"synthetic bridge"}' if self.n == 1 else 'UNUSED_DRAFT_POISON')
            return SimpleNamespace(value=dict(text=text, model_flops=30 if self.big else 10,
                generated_tokens=3, ended_with_eos=True, request_id=hashlib.sha256(json.dumps(messages).encode()).hexdigest()))
        def rank(self, payload):
            return SimpleNamespace(value=dict(ranked_paragraph_ids=[d['id'] for d in payload['paragraphs']][::-1],
                                               estimated_linear_attention_flops=5))
    fixtures = []
    for count in (0, 3, 5, 8, 20, 25):
        docs = [dict(doc_id=str(i), title=f'Doc {i}', body=f'Synthetic evidence {i}.', score=float(i)) for i in range(count)]
        def search(query, k, excluded):
            return [d for d in docs if d['doc_id'] not in excluded][:k], dict(query=query, k=k, excluded=excluded)
        result = {}
        for method in METHODS:
            b4, b8 = Backend(), Backend(True)
            result[method] = execute(dict(id='synthetic', question='Synthetic?', dataset='synthetic'), method, search, b4, b8)
            assert b4.n == (2 if method == METHODS[0] else 1) and b8.n == 1
        paid, skipped = (result[m] for m in METHODS)
        for key in ('answer', 'initial_documents', 'acquired_documents', 'protected_documents', 'final_documents',
                    'writer_parse', 'route', 'prefix_model_flops', 'ranker_model_flops', 'selected_ids', 'visibility'):
            assert paid[key] == skipped[key], key
        assert paid['calls'][-1]['_messages'] == skipped['calls'][-1]['_messages']
        assert 'UNUSED_DRAFT_POISON' not in json.dumps(skipped['calls'][-1]['_messages'])
        assert paid['total_model_flops']-skipped['total_model_flops'] == paid['draft_model_flops'] == 10
        fixtures.append(dict(available_documents=count, protected_documents=len(skipped['protected_documents']),
                             same_final_prompt=True, exactly_one_draft_call_removed=True))
    return dict(status='passed', fixtures=fixtures, learned_decisions=0)
