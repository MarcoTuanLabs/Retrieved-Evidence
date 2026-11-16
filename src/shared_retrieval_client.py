"""One client per caller thread, sharing the same resident retrieval service."""
import hashlib
import http.client
import json
import socket
import time
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


class SharedSearchClient:
    def __init__(self, port, index_sha, timeout=5.):
        self.port, self.index_sha, self.timeout = port, index_sha, timeout
        self.connection = None

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.connection = None

    def search(self, query, count, excluded, *, request_id=None):
        logical_id = request_id or uuid.uuid4().hex
        request = {'query': query, 'k': count, 'exclude_doc_ids': list(excluded)}
        errors, attempts = [], []
        started = time.perf_counter()
        for attempt in range(2):
            tick = time.perf_counter()
            try:
                if self.connection is None:
                    self.connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
                if self.connection.sock is None:
                    self.connection.connect()
                    self.connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.connection.request('POST', '/search', canonical(request).encode(),
                    {'Content-Type': 'application/json', 'X-Request-ID': f'{logical_id}:attempt{attempt}'})
                response = self.connection.getresponse()
                body = response.read()
                result = json.loads(body)
                attempts.append({'attempt': attempt, 'seconds': time.perf_counter() - tick,
                                 'status': response.status, 'bytes': len(body)})
                if response.status != 200:
                    raise RuntimeError('HTTP ' + str(response.status))
                if result['index_sha256'] != self.index_sha:
                    raise ValueError('Unexpected corpus hash')
                if any(set(d) != {'doc_id', 'title', 'body', 'score'} for d in result['documents']):
                    raise ValueError('Unexpected evidence schema')
                elapsed = time.perf_counter() - started
                capture = {'request': request, 'documents': result['documents'], 'fts_query': result['fts_query']}
                return result['documents'], {'request': request, '_capture': capture,
                    'logical_request_id': logical_id, 'request_id': result['request_id'],
                    'http_seconds': elapsed, 'attempts': len(attempts), 'attempt_records': attempts,
                    'errors': errors, 'query_wall_ms': result['query_wall_ms'],
                    'query_cpu_ms': result['query_cpu_ms'], 'response_bytes': len(body),
                    'response_sha256': hashlib.sha256(canonical(capture).encode()).hexdigest(),
                    'no_hit': not result['documents'], 'terminal_failure': False,
                    'failed_attempt_server_cpu_unknown': bool(errors)}
            except (OSError, http.client.HTTPException, RuntimeError, json.JSONDecodeError) as error:
                errors.append(type(error).__name__ + ': ' + str(error))
                if len(attempts) <= attempt:
                    attempts.append({'attempt': attempt, 'seconds': time.perf_counter() - tick,
                                     'error': errors[-1]})
                self.close()
        return [], {'request': request, '_capture': None, 'logical_request_id': logical_id,
                    'http_seconds': time.perf_counter() - started, 'attempts': len(attempts),
                    'attempt_records': attempts, 'errors': errors, 'query_wall_ms': None,
                    'query_cpu_ms': None, 'response_bytes': 0, 'response_sha256': None,
                    'no_hit': True, 'terminal_failure': True, 'failed_attempt_server_cpu_unknown': True}
