"""Pinned actual GPU backend. Call load/answer/rank only on the stage worker."""
import hashlib
import json
import threading
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig, StaticCache

from costs_v2 import DenseForwardCost
from deadline_work_queue import StageResult
from qasper_runtime_panel import static_generation_work
from semantic_paragraph_batched import BatchedSemanticParagraphRanker
from shared_retrieval_client import canonical

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
CAPACITIES = (1024, 2048, 4096, 8192, 16384)


class SynchronizedTokenizer:
    """Rust tokenizer configuration is mutable; serialize shared tokenization."""
    def __init__(self, tokenizer):
        self.raw, self.lock = tokenizer, threading.RLock()

    def __getattr__(self, key):
        value = getattr(self.raw, key)
        if not callable(value):
            return value
        def call(*args, **kwargs):
            with self.lock:
                return value(*args, **kwargs)
        return call


class CudaBackend:
    kind = 'actual_qwen4b_cuda_bf16_sdpa_static_compiled'

    def __init__(self, cache, reranker):
        self.cache_path, self.reranker_path = cache, reranker
        self.tokenizer = SynchronizedTokenizer(AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir=cache, local_files_only=True))
        self.owner = None

    def load(self):
        self.owner = threading.get_ident()
        tick = time.perf_counter()
        torch.manual_seed(1307)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, cache_dir=self.cache_path,
            local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda').eval()
        self.cost = DenseForwardCost.from_config_total(self.model.config.to_dict(), sum(p.numel() for p in self.model.parameters()))
        self.ranker = BatchedSemanticParagraphRanker(self.reranker_path, device='cuda', batch_size=16,
                                                   precision='bfloat16', instrument=False)
        self.caches = {c: StaticCache(config=self.model.config, max_cache_len=c) for c in CAPACITIES}
        self.compile = CompileConfig(mode='reduce-overhead', fullgraph=True)
        self.common = dict(max_new_tokens=128, do_sample=False, use_cache=True,
                           logits_to_keep=1, pad_token_id=self.tokenizer.eos_token_id)
        warm = []
        token = self.tokenizer.encode('example', add_special_tokens=False)[0]
        with torch.inference_mode():
            x = torch.full((1, 32), token, device='cuda', dtype=torch.long)
            for capacity in CAPACITIES:
                start = time.perf_counter()
                self.generate(x, capacity, max_new_tokens=2, min_new_tokens=2)
                torch.cuda.synchronize()
                assert hasattr(self.model, '_compiled_call')
                warm.append(dict(capacity=capacity, seconds=time.perf_counter() - start))
            self.ranker.rank('What is the topic?', [{'id': 'warmup', 'section': 'Example', 'text': 'A generic topic.'}])
        metadata = dict(model=MODEL, revision=REVISION, backend=self.kind, load_and_warm_seconds=time.perf_counter() - tick,
            torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
            gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory, threads=torch.get_num_threads(),
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, float32_matmul_precision=torch.get_float32_matmul_precision(),
            torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            model_config=self.model.config.to_dict(), generation_config=self.model.generation_config.to_dict(),
            tokenizer_special_tokens=self.tokenizer.special_tokens_map, cost_model=self.cost.__dict__, warmup=warm,
            model_parameters=sum(p.numel() for p in self.model.parameters()),
            ranker_parameters=sum(p.numel() for p in self.ranker.model.parameters()),
            startup_cost_scope='Separate from online requests; warmup recorded, no energy claim')
        metadata['generation_validity_check'] = 'return_dict_in_generate=True, output_logits=True; finite raw logits checked after generation; paid GPU time and memory'
        return StageResult(metadata, {'scope': 'startup and generic warmup; not an online request'})

    def generate(self, x, capacity, **overrides):
        assert threading.get_ident() == self.owner
        cache = self.caches[capacity]
        cache.reset()
        options = dict(self.common, past_key_values=cache, compile_config=self.compile, disable_compile=False)
        options.update(overrides)
        return self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), **options)

    def answer(self, messages):
        assert threading.get_ident() == self.owner
        tick = time.perf_counter()
        with torch.inference_mode():
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            capacity = next(c for c in CAPACITIES if c >= len(ids) + self.common['max_new_tokens'])
            assert len(ids) <= 15360
            x = torch.tensor([ids], device='cuda')
            torch.cuda.reset_peak_memory_stats()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            output = self.generate(x, capacity, return_dict_in_generate=True, output_logits=True)
            # A single post-generation validity check, paid by every method. Raw
            # logits avoid confusing intentional -inf decoding masks with NaNs.
            finite = torch.stack([torch.isfinite(v).all() for v in output.logits]).all()
            end.record()
            torch.cuda.synchronize()
            if not bool(finite.item()):
                raise RuntimeError('Nonfinite raw generation logits; preserve failed stage with unknown cost')
            generated = output.sequences[0, len(ids):].cpu().tolist()
            value = dict(text=self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                prompt_tokens=len(ids), generated_tokens=len(generated), generated_ids=generated,
                input_ids_sha256=hashlib.sha256(canonical(ids).encode()).hexdigest(),
                raw_logits_finite=True, raw_logits_checked_steps=len(output.logits),
                elapsed_seconds=time.perf_counter() - tick,
                model_flops=static_generation_work(self.cost, len(ids), len(generated), capacity),
                cuda_stream_elapsed_seconds=begin.elapsed_time(end) / 1000,
                static_capacity=capacity, ended_with_eos=bool(generated and generated[-1] == self.tokenizer.eos_token_id),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                request_id=hashlib.sha256(canonical(messages).encode()).hexdigest())
        return StageResult(value, {k: value[k] for k in ('model_flops', 'cuda_stream_elapsed_seconds', 'prompt_tokens', 'generated_tokens', 'peak_allocated_bytes')})

    def rank(self, payload):
        assert threading.get_ident() == self.owner
        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            value = self.ranker.rank(payload['question'], payload['paragraphs'])
            end.record()
            torch.cuda.synchronize()
            cost = dict(model_flops=value['estimated_linear_attention_flops'],
                        cuda_stream_elapsed_seconds=begin.elapsed_time(end) / 1000,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
        return StageResult(value, cost)
