"""Frozen exploratory controls and transferred priors under a common callback."""
import hashlib

from freeze_qasper_question_rules import choose
from qasper_online_selectors import fixed_selection, messages_for, select
from qasper_prior_selectors import POLICIES as PRIOR_POLICIES, run_policy

POLICIES = PRIOR_POLICIES + ('learned_0.005', 'gated_gap_stump_oof')
CAPACITIES = (1024, 2048, 4096, 8192, 16384, 32768)


def capacity_for(prompt_tokens):
    return next(c for c in CAPACITIES if c >= prompt_tokens+64)


def fold_for(paper_id):
    return int(hashlib.sha256(('paid-depth-signal-v1:'+paper_id).encode()).hexdigest()[:8], 16) % 5


def static_generation_work(cost, prompt_tokens, generated_tokens, capacity):
    assert 0 < prompt_tokens < capacity and 0 < generated_tokens <= capacity-prompt_tokens
    extra = max(generated_tokens-1, 0)
    linear = (2*cost.transformer_parameters*prompt_tokens + 2*cost.head_parameters
              + 2*(cost.transformer_parameters+cost.head_parameters)*extra)
    attention = 4*cost.layers*cost.attention_width*capacity*(prompt_tokens+extra)
    return linear+attention


def run_panel_policy(observation, policy, ranker, tacer, answer_fn,
                     tokenizer, fitted, dynamic_cost, stump):
    if policy in PRIOR_POLICIES:
        return run_policy(observation, policy, ranker, tacer, answer_fn)
    assert set(observation) == {'question', 'title', 'paragraphs'}
    if policy == 'learned_0.005':
        # Frozen old policy still uses its original dynamic-cost prediction.
        messages, details = select(observation, policy, tokenizer, fitted, dynamic_cost, ranker)
        details['cost_prediction_runtime'] = 'original_dynamic_eager'
    else:
        assert policy == 'gated_gap_stump_oof'
        bypass = choose(observation['question'])['question_rule_top10'] == 'bm25_3'
        details = {'policy': policy, 'early_bypass': bypass, 'selector_model_flops': 0,
                   'ranking_called': not bypass, 'out_of_fold_model': True}
        if bypass:
            ids = fixed_selection(observation, 'bm25_3')
            details['chosen_mode'] = 'bm25_3'
        else:
            ranked = ranker.rank(observation['question'], observation['paragraphs'])
            ordered = ranked['ranked_paragraph_ids']
            # Training uses float64 feature values, sklearn internally float32.
            # These BF16 scores and their difference are exactly float32 here.
            assert len(ordered) >= 6
            gap = ranked['paragraph_scores'][ordered[4]]-ranked['paragraph_scores'][ordered[5]]
            prediction = stump['left'] if gap <= stump['threshold'] else stump['right']
            count = 10 if prediction > 0 else 5
            ids = ordered[:count]
            details.update(ranking=ranked, gap5_6=gap, predicted_incremental_utility=prediction,
                           selected_count=len(ids), chosen_mode='semantic_'+str(count),
                           selector_model_flops=ranked['estimated_linear_attention_flops'])
        messages = messages_for(observation, ids)
        details['selected_paragraph_ids'] = ids
    call = answer_fn(messages)
    return {'answer': call['text'], 'calls': [call], 'details': details,
            'total_model_flops': details['selector_model_flops']+call['model_flops']}
