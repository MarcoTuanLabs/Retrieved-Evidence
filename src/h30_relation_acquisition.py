"""Fixed relation-following acquisition; unchanged H29 answer/comparison rule.

Only a question, the initial retrieved documents, and a charged search callback
are available. Extra documents never leak into the unchanged fallback prompt.
"""
import re

from h29_comparator import parse_question, decide, grounded_document, director_document, normalize

MODES = ['prefix_only', 'original_expansion', 'relation_following']
PARTICLES = {'de', 'del', 'della', 'di', 'da', 'dos', 'du', 'van', 'von', 'der', 'den', 'la', 'le'}
ALLOWED_TAIL = re.compile(r'^(?:$|[.;]|,?\s*(?:and starring|and written|and produced|who\b))', re.I)


def director_name(film):
    """Extract a unique capitalized name from a visible directed-by clause.

    Conservatively reject co-directors or an unrecognized suffix. No title index
    lookup, entity dictionary, model inference or date-label access is allowed.
    """
    choices = []
    for match in re.finditer(r'\bdirected by\s+', film['body'], re.I):
        if film['body'][max(0,match.start()-3):match.start()].lower() == 'co-': return None
        tail = film['body'][match.end():match.end()+180]
        words = list(re.finditer(r'\S+', tail)); end = 0; tokens = []
        for index, word in enumerate(words):
            token = word.group(); core = token.rstrip(',.;:')
            if not core or not (core[0].isupper() or (tokens and core in PARTICLES)): break
            if not re.fullmatch(r'[^\W\d_][\w’\x27.\-]*',core,re.UNICODE): break
            tokens.append(core if token.endswith((',', ';', ':')) else token)
            end = word.start()+len(token.rstrip(',;:'))
            if token.endswith((',', ';', ':')): break
            if token.endswith('.'):
                # A single initial is part of the name only if another name token follows.
                initial = len(core)==1 or bool(re.fullmatch(r'(?:[^\W\d_]\.)+[^\W\d_]',core))
                following = words[index+1].group() if index+1<len(words) else ''
                if not (initial and following and following[0].isupper()):
                    end -= 1; tokens[-1] = tokens[-1].rstrip('.'); break
            if len(tokens)>7: return None
        if not tokens or len(tokens)>7: return None
        name = tail[:end].strip()
        if not ALLOWED_TAIL.match(tail[end:].lstrip()): return None
        if name.split()[-1] in PARTICLES: return None
        choices.append(dict(name=name,film_id=film['doc_id'],clause=film['body'][match.start():match.end()+end]))
    unique = {normalize(c['name']):c for c in choices}
    return next(iter(unique.values())) if len(unique)==1 else None


def execute(question, initial_documents, mode, search):
    assert mode in MODES and len(initial_documents)<=5
    parsed = parse_question(question)
    documents = [dict(d) for d in initial_documents]
    initial = decide(parsed,documents)
    calls,links = [],[]
    seen_queries = set()
    def acquire(query, k, excluded, reason):
        key = normalize(query)
        if key in seen_queries: return
        seen_queries.add(key)
        docs, receipt = search(query,k,excluded)
        assert receipt['query']==query and receipt['k']==k and receipt['excluded']==excluded
        calls.append(dict(reason=reason,receipt=receipt))
        by_id = {d['doc_id']:d for d in documents}
        for d in docs:
            if d['doc_id'] in by_id:
                assert all(by_id[d['doc_id']][f]==d[f] for f in ['title','body'])
            else: documents.append(dict(d));by_id[d['doc_id']]=d
    if parsed and not initial['accepted']:
        if mode=='original_expansion':
            acquire(question,20,[d['doc_id'] for d in initial_documents],'original_question_next20')
        elif mode=='relation_following':
            # Entity names are in the question; director names must be in visible film text.
            for entity in parsed['entities']:
                if grounded_document(entity,documents) is None:
                    acquire(entity,5,[],'missing_question_entity')
            if parsed['kind'].startswith('director_'):
                for entity in parsed['entities']:
                    film = grounded_document(entity,documents)
                    if film is None: continue
                    if director_document(film,documents) is not None: continue
                    link = director_name(film)
                    if link is None: continue
                    links.append(link)
                    if grounded_document(link['name'],documents) is None:
                        acquire(link['name'],5,[],'missing_visible_director')
    decision = decide(parsed,documents)
    assert len(calls)<=4 and len(documents)<=25
    if mode=='prefix_only': assert not calls
    if mode=='original_expansion': assert len(calls)<=1
    return dict(parsed=parsed,initial_decision=initial,decision=decision,links=links,
        acquisition_calls=calls,final_documents=documents,initial_document_ids=[d['doc_id'] for d in initial_documents])
