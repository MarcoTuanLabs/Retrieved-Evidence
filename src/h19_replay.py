"""Read-only causal replay and independent cost reconstruction for H19."""
import copy
import hashlib
import json
import math
from types import SimpleNamespace
from h19_pipeline import execute


class Replay:
    def __init__(self, requests=None, tokenizers=None, costs=None, ranker_tokenizer=None, ranker_config=None):
        self.requests, self.tokenizers, self.costs = requests, tokenizers, costs
        self.ranker_tokenizer, self.ranker_config = ranker_tokenizer, ranker_config
        self.search_cache, self.input_cache, self.rank_cache = {}, {}, {}
        self.calls = self.rankings = 0
        self.max_search_difference = 0.

    def run(self, observation, method, captured, db, dataset_key, skip_captured_draft=False):
        from shared_retrieval_client import canonical
        from local_retrieval_server import search
        from qasper_runtime_panel import static_generation_work
        owner = self
        queue = copy.deepcopy(captured['calls'])
        if skip_captured_draft:
            assert [c['role'] for c in queue] == ['query_writer', 'draft4', 'answer8']
            queue = [c for c in queue if c['role'] != 'draft4']
        expected_roles = ['query_writer', 'draft4', 'answer8'] if method == 'paid_draft_fixed10' else ['query_writer', 'answer8']
        assert [c['role'] for c in queue] == expected_roles
        acquired = iter(captured['acquisition_calls'])

        class Backend:
            def __init__(self, role):
                self.role = role
                if owner.tokenizers:
                    self.tokenizer = owner.tokenizers[role]
                else:
                    raise AssertionError('Actual tokenizer required to reproduce message truncation')
            def answer(self, messages):
                call = queue.pop(0)
                body = canonical(messages).encode()
                rid = hashlib.sha256(body).hexdigest()
                assert rid == call['request_id']
                if owner.requests:
                    assert (owner.requests/(rid+'.json')).read_bytes() == body
                key = self.role, rid
                if key not in owner.input_cache:
                    owner.input_cache[key] = self.tokenizer.apply_chat_template(messages, tokenize=True,
                        add_generation_prompt=True, enable_thinking=False)
                tokens = owner.input_cache[key]
                assert len(tokens) == call['prompt_tokens'] <= 15360
                assert hashlib.sha256(canonical(tokens).encode()).hexdigest() == call['input_ids_sha256']
                capacity = next(c for c in (1024, 2048, 4096, 8192, 16384) if c >= len(tokens)+128)
                assert capacity == call['static_capacity']
                assert len(call['generated_ids']) == call['generated_tokens'] <= 128
                assert self.tokenizer.decode(call['generated_ids'], skip_special_tokens=True).strip() == call['text']
                assert call['raw_logits_finite'] and call['raw_logits_checked_steps'] == call['generated_tokens']
                assert call['ended_with_eos'] == bool(call['generated_ids'] and call['generated_ids'][-1] == self.tokenizer.eos_token_id)
                if owner.costs:
                    assert static_generation_work(owner.costs[self.role], len(tokens), call['generated_tokens'], capacity) == call['model_flops']
                owner.calls += 1
                return SimpleNamespace(value=call)
            def rank(self, payload):
                rank = captured['ranking']
                assert set(rank['ranked_paragraph_ids']) == {p['id'] for p in payload['paragraphs']}
                assert len(rank['ranked_paragraph_ids']) == len(payload['paragraphs'])
                assert all(math.isfinite(s) for s in rank['window_scores'])
                scores = {p['id']: -math.inf for p in payload['paragraphs']}
                assert len(rank['windows']) == len(rank['window_scores'])
                for window, score in zip(rank['windows'], rank['window_scores']):
                    scores[window['paragraph_id']] = max(scores[window['paragraph_id']], score)
                assert scores == rank['paragraph_scores']
                assert sorted(scores, key=lambda p: (-scores[p], p)) == rank['ranked_paragraph_ids']
                if owner.ranker_tokenizer:
                    from semantic_paragraph_baseline import SemanticParagraphRanker
                    key = canonical(payload)
                    if key not in owner.rank_cache:
                        pairs, windows, details = SemanticParagraphRanker.windows(
                            SimpleNamespace(tokenizer=owner.ranker_tokenizer), payload['question'], payload['paragraphs'])
                        shapes = [[len(pairs[o:o+16]), max(len(x['input_ids']) for x in pairs[o:o+16])]
                                  for o in range(0, len(pairs), 16)]
                        owner.rank_cache[key] = windows, details, shapes
                    windows, details, shapes = owner.rank_cache[key]
                    assert windows == rank['windows'] and details == rank['details'] and shapes == rank['batch_shapes']
                    cfg = owner.ranker_config
                    hidden, intermediate, layers = (cfg[k] for k in ('hidden_size', 'intermediate_size', 'num_hidden_layers'))
                    flops = sum(2*b*l*layers*(4*hidden**2+2*hidden*intermediate)
                        +2*b*(hidden**2+hidden*len(cfg['id2label']))+4*layers*b*l*l*hidden for b,l in shapes)
                    assert flops == rank['estimated_linear_attention_flops']
                    assert sum(b*l for b,l in shapes) == rank['padded_tokens']
                owner.rankings += 1
                return SimpleNamespace(value=copy.deepcopy(rank))

        def retrieve(query, k, excluded):
            old = next(acquired)
            assert old['query'] == query and old['k'] == k and old['excluded'] == excluded
            key = dataset_key, query, k, tuple(excluded)
            if key not in owner.search_cache:
                owner.search_cache[key] = search(db, query, k, excluded)
            docs, fts = owner.search_cache[key]
            assert fts == old['fts_query'] and len(docs) == len(old['documents'])
            for actual, recorded in zip(docs, old['documents']):
                assert {k:v for k,v in actual.items() if k != 'score'} == {k:v for k,v in recorded.items() if k != 'score'}
                difference = abs(actual['score']-recorded['score'])
                owner.max_search_difference = max(owner.max_search_difference, difference)
                assert difference <= 1e-12
            return copy.deepcopy(old['documents']), old

        result = execute(observation, method, retrieve, Backend('qwen4b'), Backend('qwen8b'))
        assert not queue and next(acquired, None) is None
        return result


EXACT_FIELDS = ['answer', 'action', 'events', 'initial_documents', 'acquired_documents',
    'protected_documents', 'final_documents', 'writer_parse', 'route', 'decision_observation',
    'prefix_model_flops', 'draft_model_flops', 'ranker_model_flops', 'head_flops',
    'total_model_flops', 'selected_ids', 'visibility', 'draft_skipped']
