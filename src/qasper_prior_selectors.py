"""Common-observation, common-renderer comparison for transferred prior policies.

All costs of this function and its answer callback belong inside the caller's
request timer. A cache of captured rankings is only a functional test fixture.
"""
from dataclasses import dataclass

from build_qasper_answerer_pilot import SYSTEM
from freeze_qasper_question_rules import choose
from prior_context_baselines import adaptive_k_official_count
from qasper_online_selectors import fixed_selection


@dataclass(frozen=True)
class ContextDocument:
    doc_id: str
    text: str


POST_RANKING_POLICIES = ('semantic_5', 'semantic_10', 'adaptive_k_all', 'adaptive_k_top10', 'tacer')
POLICIES = ('full', 'bm25_10', 'question_rule_top10') + POST_RANKING_POLICIES + tuple(
    'gated_'+p for p in POST_RANKING_POLICIES)


def messages_for_documents(observation, documents):
    """Preserve policy-produced text; common source order for original documents.

    Upstream composite briefs retain their own internal order. They cannot be
    expanded back into full original paragraphs by looking up their identifiers.
    """
    position = {p['id']: i for i, p in enumerate(observation['paragraphs'])}
    ordered = sorted(enumerate(documents),
                     key=lambda x: (position.get(x[1].doc_id, len(position)), x[0]))
    evidence = '\n\n'.join(doc.text for _, doc in ordered)
    user = ('Paper: '+observation['title']+'\n\nPaper text:\n'+evidence+
            '\n\nQuestion: '+observation['question'])
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]


def run_policy(observation, policy, ranker, tacer, answer_fn):
    """answer_fn(messages) returns actual text, tokens, elapsed time and work."""
    assert policy in POLICIES and set(observation) == {'question', 'title', 'paragraphs'}
    paragraphs = {p['id']: p for p in observation['paragraphs']}
    calls = []

    def generate_docs(documents, config=None):
        messages = messages_for_documents(observation, documents)
        result = answer_fn(messages)
        calls.append(result)
        return result

    def original_docs(ids):
        return [ContextDocument(pid, paragraphs[pid]['section']+'\n'+paragraphs[pid]['text']) for pid in ids]

    gated = policy.startswith('gated_')
    post_policy = policy.removeprefix('gated_')
    bypass = gated and choose(observation['question'])['question_rule_top10'] == 'bm25_3'
    detail = {'policy': policy, 'gated': gated, 'early_bypass': bypass, 'selector_model_flops': 0}
    if bypass or policy in ('full', 'bm25_10', 'question_rule_top10'):
        mode = 'bm25_3' if bypass else policy
        if mode == 'question_rule_top10':
            mode = choose(observation['question'])['question_rule_top10']
        ids = fixed_selection(observation, mode)
        answer = generate_docs(original_docs(ids))['text']
        detail.update(selected_paragraph_ids=ids, chosen_mode=mode, ranking_called=False)
    else:
        # Nothing reads a ranking feature until the bypass decision has been made.
        ranked = ranker.rank(observation['question'], observation['paragraphs'])
        detail.update(ranking_called=True, ranking=ranked,
                      selector_model_flops=ranked['estimated_linear_attention_flops'])
        if post_policy == 'tacer':
            result = tacer.run(observation, ranked, generate_docs)
            answer = result['answer']
            detail['tacer'] = result
        else:
            ids = ranked['ranked_paragraph_ids']
            if post_policy.startswith('semantic_'):
                count = int(post_policy.split('_')[1])
            else:
                scores = [ranked['paragraph_scores'][pid] for pid in ids]
                count = adaptive_k_official_count(scores, pool_limit=10 if post_policy == 'adaptive_k_top10' else None)
            ids = ids[:count]
            answer = generate_docs(original_docs(ids))['text']
            detail.update(selected_paragraph_ids=ids, selected_count=len(ids))
    assert calls
    return {'answer': answer, 'calls': calls, 'details': detail,
            'total_model_flops': detail['selector_model_flops']+sum(c['model_flops'] for c in calls)}
