"""FIFO stage execution with deadlines and an append-only terminal work ledger.

An expired caller never cancels an already running operation. The worker records
its result and cost before accepting the next operation. One worker can therefore
own a model and its mutable KV cache without concurrent cache mutation.
"""
from concurrent.futures import Future, wait
from dataclasses import dataclass
import json
import math
from pathlib import Path
import queue
import threading
import time


class StageDeadline(TimeoutError):
    pass


class StageExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageResult:
    value: object
    # The operation supplies its measured/estimated cost fields, with scope labels.
    cost: dict


@dataclass(frozen=True)
class Ticket:
    future: Future
    deadline: float
    stage_id: str

    def result(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not wait([self.future], timeout=remaining).done:
            # Deliberately do NOT call Future.cancel(): running work remains paid.
            raise StageDeadline(self.stage_id)
        if time.monotonic() > self.deadline:
            raise StageDeadline(self.stage_id)
        return self.future.result()


class DeadlineWorkQueue:
    def __init__(self, ledger_path):
        self.ledger_path = Path(ledger_path)
        self.ledger = self.ledger_path.open('x')
        self.pending = queue.Queue()
        self.lifecycle_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.stage_ids = set()
        self.closed = False
        self.worker = threading.Thread(target=self._work, name='single-stage-worker', daemon=False)
        self.worker.start()

    def _record(self, row):
        with self.log_lock:
            self.ledger.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            self.ledger.flush()

    def submit(self, request_id, stage_id, operation, deadline, fn, metadata=None):
        if not all(isinstance(s, str) and s for s in (request_id, stage_id, operation)):
            raise ValueError('nonempty request/stage/operation IDs required')
        if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise ValueError('finite monotonic deadline required')
        # Copy and validate metadata now so later caller mutation cannot affect the ledger.
        frozen = json.loads(json.dumps(metadata or {}, allow_nan=False))
        if not isinstance(frozen, dict):
            raise ValueError('metadata must be an object')
        forbidden = {'gold_answer', 'gold', 'label', 'support_titles', 'reference_answer'}
        if forbidden.intersection(frozen):
            raise ValueError('evaluator fields are not scheduler metadata')
        future = Future()
        row = dict(request_id=request_id, stage_id=stage_id, operation=operation,
            metadata=frozen, submitted_monotonic=time.monotonic(), deadline_monotonic=deadline)
        with self.lifecycle_lock:
            if self.closed:
                raise RuntimeError('worker closed')
            if stage_id in self.stage_ids:
                raise ValueError('duplicate stage ID')
            self.stage_ids.add(stage_id)
            if row['submitted_monotonic'] >= deadline:
                row.update(status='expired_before_submit', executed=False,
                    cost={'model_work_estimate': 0, 'scope': 'operation never started'},
                    cost_complete=True, queue_seconds=0, execution_wall_seconds=0,
                    worker_thread_cpu_seconds=0, finished_monotonic=time.monotonic())
                self._record(row)
                future.set_exception(StageDeadline(stage_id))
            else:
                self.pending.put((row, future, fn))
        return Ticket(future, deadline, stage_id)

    def _work(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                row, future, fn = item
                started = time.monotonic()
                row['queue_seconds'] = started - row['submitted_monotonic']
                if started >= row['deadline_monotonic']:
                    row.update(status='expired_queued', executed=False,
                        cost={'model_work_estimate': 0, 'scope': 'operation never started'},
                        cost_complete=True, execution_wall_seconds=0,
                        worker_thread_cpu_seconds=0, finished_monotonic=started)
                    self._record(row)
                    future.set_exception(StageDeadline(row['stage_id']))
                    continue
                row.update(executed=True, started_monotonic=started)
                cpu = time.thread_time()
                result, error = None, None
                try:
                    result = fn()
                    if not isinstance(result, StageResult) or not isinstance(result.cost, dict):
                        raise TypeError('operation must return StageResult(value, cost=dict)')
                    # This also rejects non-serializable/NaN results before publishing success.
                    frozen = json.loads(json.dumps(dict(value=result.value, cost=result.cost),
                                                   ensure_ascii=False, allow_nan=False))
                    row.update(result=frozen['value'], cost=frozen['cost'], cost_complete=True)
                except BaseException as caught:
                    error = StageExecutionError(type(caught).__name__ + ': ' + str(caught))
                    row.update(error=str(error), cost=None, cost_complete=False)
                finished = time.monotonic()
                row.update(finished_monotonic=finished, execution_wall_seconds=finished - started,
                    worker_thread_cpu_seconds=time.thread_time() - cpu,
                    finished_after_deadline=finished > row['deadline_monotonic'])
                if error is not None:
                    row['status'] = 'failed'
                else:
                    row['status'] = 'completed_late' if row['finished_after_deadline'] else 'completed_on_time'
                # Publish the terminal record before waking a successful caller. Keeping
                # results in this ledger preserves the cost of abandoned in-flight work.
                self._record(row)
                if error is not None:
                    future.set_exception(error)
                elif row['finished_after_deadline']:
                    future.set_exception(StageDeadline(row['stage_id']))
                else:
                    future.set_result(result.value)
            finally:
                self.pending.task_done()

    def drain(self):
        self.pending.join()

    def close(self):
        if threading.current_thread() is self.worker:
            raise RuntimeError('worker cannot close itself')
        with self.lifecycle_lock:
            if self.closed:
                return
            self.closed = True
            self.pending.put(None)
        self.worker.join()
        self.ledger.close()

