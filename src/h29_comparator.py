"""One abstaining chronology rule. Inputs: question and actually retrieved documents.

No labels, dataset identifiers, model outputs, searches or file access in this module.
Dates are intervals: an incomplete date can only decide a non-overlapping comparison.
"""
import calendar
import datetime
import re
import unicodedata


MONTHS = {name.casefold(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_PATTERN = '(?:' + '|'.join(MONTHS) + ')'
STRUCTURE = re.compile(r'\b(both|either|same|different|older|younger|earlier|later|more|less)\b|\sor\s', re.I)


def normalize(text):
    text = unicodedata.normalize('NFKD', text).casefold()
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in text if not unicodedata.combining(c)).split())


def base_title(text):
    return re.sub(r'\s*\([^()]*\)\s*$', '', text).strip()


def parse_question(question):
    q = question.strip().rstrip('?').strip()
    was = re.fullmatch(r'Was (.+) born (first|earlier|later)', q, re.I)
    if was:
        names, direction = was.groups(); kind = 'birth'
    elif ', ' in q:
        head, names = q.split(', ', 1)
        direct = re.fullmatch(r'Who (?:was born|died|is) (first|earlier|later|older|younger)', head, re.I)
        film = re.fullmatch(r'Which film (?:was released|came out) (first|earlier|later|more recently)', head, re.I)
        director = re.fullmatch(r'Which film (?:has the director(?: who)?|whose director) (?:was |is )?(born|died|older|younger)(?: (first|earlier|later|than the other))?', head, re.I)
        if direct:
            direction = direct[1].lower(); kind = 'death' if 'died' in head.lower() else 'birth'
            if ('is ' in head.lower()) != (direction in ['older', 'younger']): return None
        elif film:
            direction = film[1].lower(); kind = 'film_release'
        elif director:
            attribute, order = director.groups(); attribute = attribute.lower()
            if attribute in ['born', 'died'] and order not in ['first', 'earlier', 'later']: return None
            if attribute in ['older', 'younger'] and order not in [None, 'than the other']: return None
            direction = order if attribute in ['born', 'died'] else attribute
            kind = 'director_death' if attribute == 'died' else 'director_birth'
        else:
            return None
    else:
        return None
    pair = re.split(r'\s+or\s+', names, flags=re.I)
    if len(pair) != 2 or not all(x.strip() for x in pair): return None
    entities = [x.strip() for x in pair]
    if normalize(entities[0]) == normalize(entities[1]): return None
    return dict(kind=kind, entities=entities, choose='min' if direction.lower() in ['first', 'earlier', 'older'] else 'max')


def grounded_document(entity, documents):
    exact = [d for d in documents if normalize(d['title']) == normalize(entity)]
    matches = exact or [d for d in documents if normalize(base_title(d['title'])) == normalize(base_title(entity))]
    if len(matches) != 1: return None
    doc = matches[0]
    if not normalize(doc['body']).startswith(normalize(base_title(doc['title']))): return None
    return doc


def date_interval(text):
    """Read one date at the beginning. Never choose among multiple candidate years."""
    text = text.strip().casefold()
    if re.search(r'\b(?:circa|about|before|after|bc|bce|uncertain)\b|\bc\s*\.', text): return None
    patterns = [
        (rf'^(\d{{1,2}})\s+({MONTH_PATTERN})\s+(\d{{3,4}})\b', 'dmy'),
        (rf'^({MONTH_PATTERN})\s+(\d{{1,2}}),?\s+(\d{{3,4}})\b', 'mdy'),
        (rf'^({MONTH_PATTERN})\s+(\d{{3,4}})\b', 'my'),
        (r'^(\d{3,4})\b', 'y')]
    for pattern, form in patterns:
        match = re.match(pattern, text, re.I)
        if not match: continue
        # A second year or an alternative date makes this an ambiguous extraction.
        if re.search(r'\b\d{3,4}\b|\bor\b', text[match.end():]): return None
        try:
            if form == 'dmy': day, month, year = int(match[1]), MONTHS[match[2]], int(match[3])
            elif form == 'mdy': month, day, year = MONTHS[match[1]], int(match[2]), int(match[3])
            elif form == 'my': month, year = MONTHS[match[1]], int(match[2]); day = None
            else: year = int(match[1]); month = day = None
            lo = datetime.date(year, month or 1, day or 1)
            hi = datetime.date(year, month or 12, day or (calendar.monthrange(year, month)[1] if month else 31))
        except ValueError:
            return None
        return dict(lo=lo.isoformat(), hi=hi.isoformat(), text=text[:match.end()])
    return None


def biography_date(doc, attribute):
    body = doc['body']
    intro = re.split(r'\b(?:is|was)\b', body[:600], maxsplit=1, flags=re.I)[0]
    dates = []
    for content in re.findall(r'\(([^()]*)\)', intro):
        if not re.search(r'\b\d{3,4}\b', content): continue
        born = re.match(r'\s*born\s+(.+)', content, re.I)
        if born:
            if attribute == 'birth':
                value = date_interval(born[1])
                if value: dates.append(value)
        else:
            parts = re.split(r'\s*[–—]\s*|\s+-\s+', content)
            if len(parts) != 2: continue
            values = [date_interval(part) for part in parts]
            if all(values) and values[0]['lo'] < values[1]['hi']:
                dates.append(values[0 if attribute == 'birth' else 1])
    unique = {(v['lo'], v['hi']): v for v in dates}
    return next(iter(unique.values())) if len(unique) == 1 else None


def release_date(doc):
    # Film year in an identity clause, not a year mentioned elsewhere in the article.
    body = doc['body']; values = []
    lead = re.split(r'(?<=[.!?])\s*', body, maxsplit=1)[0]
    for pattern in [r'\bis\s+(?:a|an)\s+(\d{4})\b[^.!?]{0,120}\bfilm\b',
                    r'\(\s*(\d{4})\s*\)\s+is\s+(?:a|an)\b[^.!?]{0,120}\bfilm\b']:
        for match in re.finditer(pattern, lead, re.I):
            value = date_interval(match[1])
            if value: values.append(value)
    unique = {(v['lo'], v['hi']): v for v in values}
    return next(iter(unique.values())) if len(unique) == 1 else None


def director_document(film, documents):
    """Require the director's retrieved title at the start of a directed-by clause."""
    selected = []
    for match in re.finditer(r'\bdirected by\s+', film['body'], re.I):
        tail = film['body'][match.end():match.end()+180]
        candidates = []
        for doc in documents:
            title = base_title(doc['title'])
            # Keep punctuation in the matching expression so the suffix remains inspectable.
            expression = r'\s*'.join(re.escape(p) for p in title.split())
            found = re.match(expression + r'(?=$|[^\w])', tail, re.I)
            if not found: continue
            rest = tail[found.end():].lstrip()
            if not re.match(r'^(?:$|[.;]|,?\s*(?:and starring|and written|and produced|who\b))', rest, re.I): continue
            person = grounded_document(doc['title'], documents)
            if person: candidates.append(person)
        if len(candidates) != 1: return None
        selected.extend(candidates)
    unique = {d['doc_id']: d for d in selected}
    return next(iter(unique.values())) if len(unique) == 1 else None


def decide(parsed, documents):
    if parsed is None: return dict(accepted=False, reason='unsupported_question', evidence=[])
    evidence = []
    for entity in parsed['entities']:
        doc = grounded_document(entity, documents)
        if doc is None: return dict(accepted=False, reason='entity_missing_or_ambiguous', evidence=evidence)
        chain = [doc['doc_id']]
        kind = parsed['kind']
        if kind.startswith('director_'):
            doc = director_document(doc, documents)
            if doc is None: return dict(accepted=False, reason='director_link_missing_or_ambiguous', evidence=evidence)
            chain.append(doc['doc_id']); kind = kind.removeprefix('director_')
        value = release_date(doc) if kind == 'film_release' else biography_date(doc, kind)
        if value is None: return dict(accepted=False, reason='date_missing_or_ambiguous', evidence=evidence)
        evidence.append(dict(entity=entity, document_ids=chain, value=value))
    a, b = [e['value'] for e in evidence]
    if a['hi'] < b['lo']: early = 0
    elif b['hi'] < a['lo']: early = 1
    else: return dict(accepted=False, reason='overlapping_dates', evidence=evidence)
    winner = early if parsed['choose'] == 'min' else 1-early
    return dict(accepted=True, reason='explicit_disjoint_dates', answer=parsed['entities'][winner], evidence=evidence)
