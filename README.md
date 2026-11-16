# Retrieved-Evidence

Code and measured results for **Using Retrieved Evidence Before Searching Again**.

A retrieval-augmented question-answering system can find the right facts and still answer incorrectly, while further searches and model calls add cost. We check whether the initial retrieved documents already support an explicit birth-date, death-date, or film-release comparison. The rule matches entities to documents, represents incomplete dates as intervals, answers only when the ordering is unambiguous, and otherwise uses the original query-rewrite and reader pipeline. An optional extension follows missing entity or film-to-director links with at most four additional searches. The experiments separate evidence acquisition, evidence selection, deterministic comparison, and removal of an unused draft generation.

## Main results

The frozen confirmation uses 800 new 2WikiMultiHopQA questions, balanced across four question types, with a Qwen3-4B query writer and Qwen3-8B reader.

| Method | Exact answers / 800 | F1 (%) | Mean latency (ms) | Estimated neural computation vs. rewrite |
|---|---:|---:|---:|---:|
| Direct search | 276 | 44.41 | 652 | 69.26% |
| Rewrite baseline | 356 | 54.52 | 1002 | 100.00% |
| Check first | 375 | 57.03 | 928 | 93.96% |
| Check + links | 384 | 58.08 | 889 | 91.86% |

The primary check-first comparison improves exact match by **2.375 percentage points** (paired 95% interval: **1.25 to 3.50**), reduces estimated neural computation by **6.04%**, and reduces mean latency by **7.39%**. On the 58 questions accepted from initial evidence, a reader given that same initial evidence answers 43 correctly, the rule answers 58, and a reader given only the complete documents cited by the rule answers 56. These are selected, limited-coverage comparisons; the rules do not establish general reasoning or broad cross-dataset transfer.

In a separate 200-question MuSiQue test, removing a draft that the final reader never consumes preserves every final output token while saving **30.03%** of estimated neural computation and **42.41%** of mean latency. Estimated computation is a FLOP model, not measured energy or a billing claim. A repeated full rewrite baseline scored 355/800 versus 356/800 originally; both are retained in the released results.

## Run the lightweight checks

Python 3.11 or newer is recommended. These commands use synthetic fixtures or already recorded outcomes; they do not load language models or start paid compute.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/verify_release.py
.venv/bin/python scripts/check_pipelines.py
.venv/bin/python scripts/summarize_results.py
```

`verify_release.py` checks the SHA-256 of every original file against the extraction manifest and parses all released Python source. `check_pipelines.py` runs the original comparator fixtures and original pipeline self-checks, including abstention, fallback prompt parity, complete cost accounting, and removal of the unused draft. `summarize_results.py` recomputes headline aggregates from the released per-question scored rows and checks them against the original summaries.

## Use the evidence rule

```python
import sys
sys.path.insert(0, "src")
from h29_comparator import parse_question, decide

documents = [
    {"doc_id": "a", "title": "Ada Blue", "body": "Ada Blue (born 1950) is a director."},
    {"doc_id": "b", "title": "Ben Green", "body": "Ben Green (born 1960) is a director."},
]
result = decide(parse_question("Who was born earlier, Ada Blue or Ben Green?"), documents)
assert result["accepted"] and result["answer"] == "Ada Blue"
```

For bounded acquisition, use `h30_relation_acquisition.execute(question, initial_documents, mode, search)`, where `mode` is `prefix_only` or `relation_following`. The search callback returns `(documents, receipt)`; receipts retain the query, requested count, excluded IDs, and retrieval costs. For the complete baseline and evidence-check policies, see `h31_pipeline.execute` and its self-check for the backend interface.

## Code map

| File | Role |
|---|---|
| `src/h29_comparator.py` | Conservative entity grounding, interval dates, and abstaining comparison |
| `src/h30_relation_acquisition.py` | Bounded entity and director-link retrieval |
| `src/h31_pipeline.py` | Direct, rewrite, check-first, and check-with-links policies |
| `src/h25_pipeline.py` | Shared query-rewrite, ranking, and reader operations |
| `src/h19_pipeline.py` | With-draft and without-draft controlled pipelines |
| `src/run_h31_confirmation.py`, `src/run_h31_resume.py` | Original confirmation execution and continuation |
| `src/run_h32_supplements.py` | Original baseline repeat and fixed-evidence reader conditions |
| `src/run_h19_baseline.py` | Original draft-removal experiment runner |
| `src/analyze_h*_*.py`, `src/paired_inference.py` | Original replay, scoring, and paired inference code |
| `src/s4_cuda_backend.py`, `src/local_retrieval_server.py` | Original model execution and SQLite retrieval |
| `experiments/` | Original result summaries, per-question scores, contracts, and configurations |
| `data/h31_confirmation/observations.jsonl` | The 800 confirmation question IDs and questions |
| `provenance/original_files.json` | Original relative paths and SHA-256 hashes |

The historical H19/H25/H29/H30/H31/H32 names are retained to make the source match the experiments. Imported support modules are included, even where they belong to earlier development work.

## Full GPU experiment reproduction

This is a source-and-results release, not the complete multi-gigabyte experiment archive. Original source files are byte-identical; modules formerly under `payload/native/src` are collected under `src`, with their original paths recorded in the manifest. The original runners expect a prepared payload containing `config.json`, `manifest.json`, observations, model files, a reranker, and a SQLite retrieval index. The analysis scripts additionally expect full token/prompt captures, sealed evaluation labels, and original run receipts. They are retained for audit and adaptation and are not standalone entry points in this compact checkout.

The original H31/H32 runs use an H100, PyTorch `2.11.0+cu128`, Transformers `4.57.1`, BF16 SDPA, TF32 disabled, greedy decoding, 128 output tokens, and a 15,360-token input cap. Remaining recorded package pins are in `requirements.txt`. Exact model revisions and original run settings are in each experiment's `original_config.json`; historical deadlines and machine paths must be adapted for a new run. Do not interpret those archived configurations as current execution instructions.

Download models and datasets separately from their official sources:

- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), revision `b968826d9c46dd6066d109eabc6255188de91218`.
- [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) reranker.
- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) and [MuSiQue](https://github.com/StonyBrookNLP/musique).

Dataset material and third-party models retain their upstream terms. Model weights, corpora, raw prompt captures, remote-machine credentials, and unrelated workspace projects are excluded. The original research workspace and its Git history are not part of this repository.
