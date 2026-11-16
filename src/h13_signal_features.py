"""Strict portable reader-routing observations; no labels, IDs or future outputs."""
import hashlib
import re

import numpy as np

HASH_DIM = 1024
STOP = frozenset("a an the of in on at to for from and or is was are were be by with which what who when where how did does do this that as it its provided information answer question".split())
REFUSAL = re.compile(r"\b(?:unknown|cannot|can't|unable|not enough|insufficient|not provided|not mentioned|no information|not specified)\b", re.I)
PRE_NAMES = ["question_words", "question_content_terms", "question_digits", "relation_words",
             "starts_who", "starts_what", "starts_which", "starts_when", "starts_where", "starts_how",
             "doc_count", "total_doc_words", "mean_doc_words", "max_doc_words", "std_doc_words",
             "q_coverage_union", "q_coverage_doc_max", "q_coverage_doc_mean", "q_coverage_doc_std",
             "q_coverage_titles", "unique_doc_terms", "duplicate_title_fraction"]
DRAFT_NAMES = ["draft_words", "draft_content_terms", "generated_tokens", "ended_with_eos",
               "refusal", "unknown_exact", "yes_no_exact", "draft_digits", "draft_sentences",
               "draft_term_evidence_coverage", "draft_term_question_overlap", "draft_phrase_in_evidence"]


def words(s):
    return re.findall(r"[a-z0-9]+", s.casefold())


def ratio(n, d):
    return n / d if d else 0.


def validate(observation, kind):
    expected = {"question", "documents"} | ({"draft"} if kind == "draft_gain" else set())
    if kind not in {"pre_gain", "draft_gain"} or set(observation) != expected:
        raise ValueError("Unexpected observed fields")
    if not isinstance(observation["question"], str) or len(observation["documents"]) != 10:
        raise ValueError("Invalid question/context")
    for d in observation["documents"]:
        if set(d) != {"title", "body"} or not all(isinstance(v, str) for v in d.values()):
            raise ValueError("Non-text evidence metadata forbidden")
    if kind == "draft_gain":
        a = observation["draft"]
        if set(a) != {"text", "generated_tokens", "ended_with_eos"}:
            raise ValueError("Unexpected draft fields")
        if not isinstance(a["text"], str) or type(a["generated_tokens"]) is not int or type(a["ended_with_eos"]) is not bool:
            raise ValueError("Invalid draft")


def vector(observation, kind):
    validate(observation, kind)
    question = observation["question"]; docs = observation["documents"]
    qw = words(question); qt = set(qw) - STOP
    h = np.zeros(HASH_DIM)
    for term in qw + [" ".join(qw[i:i+2]) for i in range(len(qw)-1)]:
        z = int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "big")
        h[z % HASH_DIM] += 1 if z & (1 << 63) else -1
    norm = np.linalg.norm(h)
    if norm: h /= norm
    dw = [words(d["title"]+" "+d["body"]) for d in docs]
    dt = [set(w)-STOP for w in dw]
    union = set().union(*dt)
    title_terms = set().union(*(set(words(d["title"])) for d in docs))
    lengths = np.array([len(w) for w in dw])
    coverage = np.array([ratio(len(qt & t), len(qt)) for t in dt])
    numeric = [len(qw),len(qt),sum(w.isdigit() for w in qw),sum(w in {"whose","that","which","where","who"} for w in qw),
               *[int(bool(qw) and qw[0]==w) for w in ["who","what","which","when","where","how"]],
               len(docs),lengths.sum(),lengths.mean(),lengths.max(),lengths.std(),
               ratio(len(qt & union),len(qt)),coverage.max(),coverage.mean(),coverage.std(),
               ratio(len(qt & title_terms),len(qt)),len(union),1-len({d["title"] for d in docs})/len(docs)]
    if kind == "draft_gain":
        a=observation["draft"];aw=words(a["text"]);at=set(aw)-STOP
        normal=" ".join(aw)
        numeric += [len(aw),len(at),a["generated_tokens"],int(a["ended_with_eos"]),
                    int(bool(REFUSAL.search(a["text"]))),int(normal=="unknown"),int(normal in {"yes","no"}),
                    sum(w.isdigit() for w in aw),len(re.findall(r"[.!?]",a["text"])),
                    ratio(len(at & union),len(at)),ratio(len(at & qt),len(at)),
                    int(bool(normal) and any((" "+normal+" ") in (" "+" ".join(w)+" ") for w in dw))]
    x=np.asarray(numeric,dtype=float)
    assert len(x)==len(PRE_NAMES)+(len(DRAFT_NAMES) if kind=="draft_gain" else 0) and np.isfinite(x).all()
    return h,x


def rules(observation):
    _,x=vector(observation,"draft_gain")
    post=dict(zip(DRAFT_NAMES,x[len(PRE_NAMES):]))
    return {"rule_refusal":bool(post["refusal"]),
            "rule_unsupported":bool(post["draft_term_evidence_coverage"]<.5),
            "rule_long_or_refusal":bool(post["refusal"] or post["generated_tokens"]>=32)}


def self_check():
    o={"question":"Who built the castle?","documents":[{"title":"Castle","body":"Ada built the castle."} for _ in range(10)]}
    h,x=vector(o,"pre_gain");assert h.shape==(1024,) and x.shape==(22,)
    for bad in [{**o,"answer":"Ada"},{**o,"id":"gold"},{**o,"draft":{"text":"Ada"}}]:
        try:vector(bad,"pre_gain")
        except ValueError:pass
        else:raise AssertionError("Forbidden field accepted")
    d={**o,"draft":{"text":"Unknown: insufficient information.","generated_tokens":5,"ended_with_eos":True}}
    assert rules(d)["rule_refusal"]
    try:vector({**d,"draft":{**d["draft"],"large_model_answer":"Ada"}},"draft_gain")
    except ValueError:pass
    else:raise AssertionError("Future field accepted")
    assert np.array_equal(vector(o,"pre_gain")[0],h)
    return {"status":"passed","strict_observation_schemas":True,"forbidden_fields_rejected":4,"pre_numeric_dimensions":22,"draft_numeric_dimensions":34,"hash_dimensions":1024}
