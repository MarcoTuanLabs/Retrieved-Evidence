"""Established MiniLM cross-encoder baseline, with explicit windows and costs.

No QASPER labels or answers are inputs. Prepared infrastructure, not a novel
controller. The caller must add reading/answer-generation and model-load costs.
"""
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SemanticParagraphRanker:
    def __init__(self, model_path, device='cpu', batch_size=16, precision='float32', instrument=False):
        self.device, self.batch_size = device, batch_size
        self.precision, self.instrument = precision, instrument
        dtype={'float32':torch.float32,'bfloat16':torch.bfloat16}[precision]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, torch_dtype=dtype,
            attn_implementation='sdpa').to(device).eval()
        assert self.model.config.model_type == 'bert'
        self.linear_flops = 0
        # Count actual padded Linear inputs; no assumed unpadded throughput.
        self.handles = ([m.register_forward_hook(self._count_linear)
                         for m in self.model.modules() if isinstance(m, torch.nn.Linear)]
                        if instrument else [])
        assert self.model.bert.pooler is not None

    def _count_linear(self, module, inputs, output):
        self.linear_flops += 2 * inputs[0].numel() * module.out_features

    def _sync(self):
        if str(self.device).startswith('cuda'): torch.cuda.synchronize()

    def windows(self, question, paragraphs):
        tok = self.tokenizer
        raw_q = tok.encode(question, add_special_tokens=False, verbose=False)
        q = raw_q[:64]
        capacity = 512 - len(q) - tok.num_special_tokens_to_add(pair=True)
        overlap = 64
        assert capacity > overlap
        batches, owners, lengths = [], [], []
        for paragraph in paragraphs:
            # Preserve section information, and slide across every paragraph token.
            ids = tok.encode(paragraph['section']+'\n'+paragraph['text'], add_special_tokens=False, verbose=False)
            lengths.append(len(ids))
            offset = 0
            while True:
                piece = ids[offset:offset+capacity]
                pair = tok.prepare_for_model(q, piece, add_special_tokens=True,
                                             return_attention_mask=True, truncation=False)
                assert len(pair['input_ids']) <= 512
                batches.append(pair)
                owners.append({'paragraph_id':paragraph['id'],'start':offset,'end':offset+len(piece)})
                if offset+capacity >= len(ids): break
                offset += capacity-overlap
        return batches, owners, {'question_tokens':len(raw_q),'question_truncated':len(raw_q)>64,
                                 'paragraph_tokens':lengths,'max_pair_tokens':512,'overlap':overlap}

    def rank(self, question, paragraphs):
        assert paragraphs and len({p['id'] for p in paragraphs})==len(paragraphs)
        self._sync(); started=time.perf_counter()
        pairs,owners,details=self.windows(question, paragraphs)
        tokenize_seconds=time.perf_counter()-started
        self.linear_flops=0
        attention_flops,padded_tokens,linear_flops=0,0,0
        scores=[];batch_shapes=[]
        with torch.inference_mode():
            for offset in range(0,len(pairs),self.batch_size):
                x=self.tokenizer.pad(pairs[offset:offset+self.batch_size],padding=True,return_tensors='pt').to(self.device)
                batch,length=x['input_ids'].shape
                padded_tokens += batch*length
                cfg=self.model.config
                h,inter,layers=cfg.hidden_size,cfg.intermediate_size,cfg.num_hidden_layers
                # BERT Q/K/V/output projections, two FFN matrices, pooler and classifier.
                # Formula is checked against hooks separately; timed serving has no hooks.
                linear_flops += 2*batch*length*layers*(4*h*h+2*h*inter)+2*batch*(h*h+h*cfg.num_labels)
                attention_flops += 4*layers*batch*length*length*h
                batch_shapes.append([batch,length])
                scores.extend(self.model(**x).logits.flatten().float().cpu().tolist())
        self._sync()
        paragraph_scores={p['id']:float('-inf') for p in paragraphs}
        for window,score in zip(owners,scores):
            pid=window['paragraph_id'];paragraph_scores[pid]=max(paragraph_scores[pid],score)
        ordered=sorted(paragraph_scores,key=lambda pid:(-paragraph_scores[pid],pid))
        elapsed=time.perf_counter()-started
        if self.instrument: assert self.linear_flops==linear_flops
        return {'ranked_paragraph_ids':ordered,'paragraph_scores':paragraph_scores,
            'window_scores':scores,'windows':owners,'tokenization_seconds':tokenize_seconds,
            'reranker_total_seconds':elapsed,'estimated_linear_attention_flops':linear_flops+attention_flops,
            'cost_scope':'All padded Linear matmuls plus two attention matmuls/layer; excludes small elementwise/embedding ops, not a measured hardware FLOP counter.',
            'padded_tokens':padded_tokens,'batch_shapes':batch_shapes,'precision':self.precision,
            'instrumentation_hooks_active':self.instrument,
            'details':details,'aggregation':'maximum score over all windows per original paragraph'}
