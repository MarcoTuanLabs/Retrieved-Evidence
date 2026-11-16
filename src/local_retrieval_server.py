"""Loopback HTTP service executing FTS5/BM25, with no question/gold lookup.

The only index fields are document ID, title, and natural text. Every POST runs
the search, even when an identical query was previously requested.
"""
import argparse
import hashlib
import json
import re
import socket
import sqlite3
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STOPWORDS = set('a an the is are was were be been being of in on at for to from by with as and or what which who when where why how did does do has have had this that these those its it his her their than then during'.split())


def query_terms(query):
    words = re.findall(r'[^\W_]+', query.lower(), re.UNICODE)
    return list(dict.fromkeys(w for w in words if w not in STOPWORDS))[:32]


def search(connection, query, k, excluded):
    terms = query_terms(query)
    if not terms:
        return [], ''
    # Quoted individual tokens prevent user text from becoming FTS/SQL syntax.
    match = ' OR '.join('"' + t.replace('"', '""') + '"' for t in terms)
    sql = 'SELECT doc_id,title,body,bm25(documents,0.0,5.0,1.0) AS score FROM documents WHERE documents MATCH ?'
    params = [match]
    if excluded:
        sql += ' AND doc_id NOT IN (' + ','.join('?' for _ in excluded) + ')'
        params.extend(excluded)
    sql += ' ORDER BY score, doc_id LIMIT ?'
    params.append(k)
    rows = connection.execute(sql, params).fetchall()
    return [dict(zip(['doc_id', 'title', 'body', 'score'], row)) for row in rows], match


class SearchServer(HTTPServer):
    def get_request(self):
        sock, addr = super().get_request()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return sock, addr


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def send_json(self, status, response):
        payload = json.dumps(response, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        started, cpu = time.perf_counter(), time.process_time()
        if self.path != '/search':
            self.send_json(404, {'error': 'unknown endpoint'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 131072:
                raise ValueError('invalid request length')
            request = json.loads(self.rfile.read(length))
            if set(request) - {'query', 'k', 'exclude_doc_ids'}:
                raise ValueError('unsupported observation field')
            query, k, excluded = request['query'], request.get('k', 5), request.get('exclude_doc_ids', [])
            if not isinstance(query, str) or len(query) > 4000 or type(k) is not int or not 1 <= k <= 20:
                raise ValueError('invalid query or k')
            if not isinstance(excluded, list) or len(excluded) > 100 or any(not isinstance(x, str) or len(x) != 64 for x in excluded):
                raise ValueError('invalid document exclusions')
            docs, match = search(self.server.db, query, k, excluded)
            result = {'documents': docs, 'fts_query': match, 'index_sha256': self.server.index_sha,
                      'query_wall_ms': (time.perf_counter() - started) * 1000,
                      'query_cpu_ms': (time.process_time() - cpu) * 1000}
            self.server.log.write(json.dumps({'created_unix': time.time(), 'request': request,
                'document_ids': [d['doc_id'] for d in docs], 'query_wall_ms': result['query_wall_ms'],
                'query_cpu_ms': result['query_cpu_ms'], 'status': 200}) + '\n')
            self.server.log.flush()
            self.send_json(200, result)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self.send_json(400, {'error': str(error)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--port', type=int, default=0)
    args = p.parse_args()
    server = SearchServer(('127.0.0.1', args.port), Handler)
    server.db = sqlite3.connect('file:' + str(args.index.resolve()) + '?mode=ro', uri=True)
    server.index_sha = hashlib.sha256(args.index.read_bytes()).hexdigest()
    server.log = args.log.open('x')
    print(json.dumps({'port': server.server_port, 'index_sha256': server.index_sha}), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        server.db.close()
        server.log.close()


if __name__ == '__main__':
    main()
