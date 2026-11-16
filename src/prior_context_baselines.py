"""Adapters for established context policies; these are not new algorithms."""
import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


def adaptive_k_official_count(descending_scores, buffer_docs=5, pool_limit=None):
    """Official gap/count semantics, transferred to supplied ranking scores.

    See the pinned reference and comparison protocol. Empty/singleton behavior
    extends the upstream function to avoid its undefined empty argmin.
    """
    scores = list(descending_scores)
    if pool_limit is not None:
        assert isinstance(pool_limit, int) and pool_limit > 0
        scores = scores[:pool_limit]
    assert isinstance(buffer_docs, int) and buffer_docs >= 0
    assert all(math.isfinite(x) for x in scores)
    assert all(a >= b for a, b in zip(scores, scores[1:]))
    n = len(scores)
    if n < 2:
        return n
    usable_gaps = n-1-int(.1*(n-1))
    # max(range, key) retains the first index when gaps tie, as torch.argmin does.
    boundary = max(range(usable_gaps), key=lambda i: scores[i]-scores[i+1])
    return min(n, boundary+1+buffer_docs)


@dataclass(frozen=True)
class ObservableQuery:
    text: str
    query_id: str = 'observed-question'

    @property
    def relevant_doc_ids(self):
        raise RuntimeError('Gold evidence is unavailable to the controller')

    @property
    def reference_answer(self):
        raise RuntimeError('Reference answers are unavailable to the controller')


class TacerReferenceAdapter:
    """Single-concurrency local callback adapter to byte-pinned upstream policy."""
    def __init__(self, source_root, manifest_path, project_root):
        self.source_root = Path(source_root).resolve()
        manifest = json.loads(Path(manifest_path).read_text())
        checked = 0
        for name, metadata in manifest['files'].items():
            path = (Path(project_root)/name).resolve()
            if path.parent == self.source_root/'adaptive_retrieval':
                assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata['sha256'], path
                checked += 1
        assert checked == 7
        package = sys.modules.get('adaptive_retrieval')
        if package is not None:
            assert str(self.source_root/'adaptive_retrieval') in list(package.__path__)
        sys.path.insert(0, str(self.source_root))
        self.module = importlib.import_module('adaptive_retrieval.llm_budget')
        assert Path(self.module.__file__).resolve() == self.source_root/'adaptive_retrieval/llm_budget.py'
        self.Document = importlib.import_module('adaptive_retrieval.data').Document
        self.busy = False

    def run(self, observation, ranking, answer_fn, sequential_budget=3):
        """answer_fn(docs, config) returns measured local answer/cost information.

        Required callback fields: text, prompt_tokens, generated_tokens,
        elapsed_seconds, model_flops. Additional metadata may hold token IDs.
        No local tokenizer estimates are substituted for actual generation costs.
        """
        assert not self.busy, 'Reference global callback supports batch1 only'
        assert set(observation) == {'question', 'title', 'paragraphs'}
        paragraphs = {p['id']: p for p in observation['paragraphs']}
        ids = ranking['ranked_paragraph_ids'][:10]
        assert ids and len(set(ids)) == len(ids)
        docs = [(self.Document(doc_id=pid,
                              text=paragraphs[pid]['section']+'\n'+paragraphs[pid]['text']),
                 ranking['paragraph_scores'][pid]) for pid in ids]
        query = ObservableQuery(observation['question'])
        config = self.module.LLMConfig(model='local-common-answerer',
            max_output_tokens=64, prompt_style='concise', dry_run=False,
            require_api_key=False, require_provider_tokens=False)
        calls = []

        def local_generate(received_query, selected_docs, call_config):
            assert received_query is query
            assert call_config.prompt_style == 'concise' and call_config.max_output_tokens == 64
            result = dict(answer_fn(selected_docs, call_config))
            required = {'text', 'prompt_tokens', 'generated_tokens', 'elapsed_seconds', 'model_flops'}
            assert required <= set(result)
            assert all(result[key] >= 0 for key in required-{'text'})
            context = [{'doc_id': d.doc_id, 'text': d.text} for d in selected_docs]
            calls.append({**result, 'context': context,
                'context_sha256': hashlib.sha256(json.dumps(context, sort_keys=True,
                                                           ensure_ascii=False).encode()).hexdigest(),
                'compression_mode': call_config.compression_mode})
            return self.module.GeneratedAnswer(text=result['text'],
                prompt_tokens=result['prompt_tokens'], completion_tokens=result['generated_tokens'],
                total_tokens=result['prompt_tokens']+result['generated_tokens'],
                token_source='local_measured', generation_time_ms=round(result['elapsed_seconds']*1000))

        def disabled_api(*args, **kwargs):
            raise RuntimeError('Only the explicit local answer callback is permitted')

        old_generate, old_api = self.module.generate_answer, self.module.call_openai_chat
        self.busy = True
        self.module.generate_answer = local_generate
        self.module.call_openai_chat = disabled_api
        try:
            combined, selected, fallback, reason, first_tokens, later_tokens = self.module.task_aware_coverage_ultra_run(
                query, docs, sequential_budget, config)
        finally:
            self.module.generate_answer, self.module.call_openai_chat = old_generate, old_api
            self.busy = False
        assert calls
        prompt_tokens = sum(c['prompt_tokens'] for c in calls)
        generated_tokens = sum(c['generated_tokens'] for c in calls)
        assert combined.prompt_tokens == prompt_tokens
        assert combined.completion_tokens == generated_tokens
        assert first_tokens+later_tokens == prompt_tokens+generated_tokens
        return {'answer': combined.text, 'calls': calls, 'fallback_used': fallback,
                'route_reason': reason, 'prompt_tokens': prompt_tokens,
                'generated_tokens': generated_tokens,
                'answerer_model_flops': sum(c['model_flops'] for c in calls),
                'answerer_elapsed_seconds': sum(c['elapsed_seconds'] for c in calls),
                'selected_doc_ids': [d.doc_id for d in selected],
                'cost_scope': 'Answer calls only; complete caller must also charge ranking, control and serialization.'}
