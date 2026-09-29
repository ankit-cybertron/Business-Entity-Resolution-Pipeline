import sys
import re
import json
import gc
from pathlib import Path
from collections import namedtuple

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from config import *
from src.utils import progress, memory_status, memory_guard, cleanup

SRC = CODE_DIR / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from normalization import normalize, name_core, phonetic, skeleton, addr_core, num_tokens, fold
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz, process

FEATURE_NAMES = [
    "name_jw","name_jw_core","name_token_sort","name_token_set","name_partial",
    "name_skeleton_sim","name_phonetic_eq","name_phonetic_jw","name_len_ratio",
    "name_common_tokens",
    "addr_jw","addr_jw_core","addr_token_sort","addr_partial",
    "addr_common_tokens","addr_num_overlap","addr_both_empty","addr_one_empty",
    "num_name_overlap","num_any_overlap",
    "name_addr_avg","name_core_skeleton_agree","country_match","city_match",
    "city_conflict","state_conflict","numeric_conflict","street_number_match",
    "postal_code_match","first_token_match","token_count_ratio","acronym_match",
    "common_prefix_len","state_match","street_name_match","postal_prefix_match",
    "name_and_addr_strong",
]
assert len(FEATURE_NAMES) == N_FEATURES

Precomp = namedtuple("Precomp", [
    "n_norm","n_core","n_ph","n_sk","a_norm","a_core","country",
    "num_name","num_addr","tok1","acronym","squashed_name",
    "city","state","street_num","postal","street_name","n_tok_count"
])

def _city(a):
    ts = [t for t in a.split() if len(t) > 3]
    return ts[-1] if ts else ""

def _state(a):
    ts = [t for t in a.split() if len(t) > 3]
    return ts[-2] if len(ts) >= 2 else ""

def _leading_num(a):
    m = re.search(r"\d+", fold(a))
    return m.group(0) if m else ""

def _postal(a):
    ms = re.findall(r"(?<!\d)\d{5,6}(?!\d)", fold(a))
    return ms[0] if ms else ""

def _street_name(a):
    for t in a.split():
        if len(t) > 2 and not t.isdigit():
            return t
    return ""

def _first_tok(n):
    ts = name_core(n).split()
    return ts[0] if ts else ""

def _acronym(nc):
    ts = [t for t in nc.split() if t]
    return "".join(t[0] for t in ts) if len(ts) >= 2 else ""

def _common_prefix(a, b):
    if not a or not b:
        return 0.0
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i / m
    return 1.0

def _tok_ov(a, b):
    ta = set(a.split()) if a else set()
    tb = set(b.split()) if b else set()
    if not ta and not tb:
        return 1.0
    d = max(len(ta), len(tb))
    return len(ta & tb) / d if d else 0.0

def _num_ov(sa, sb):
    if not sa and not sb:
        return 1.0
    d = max(len(sa), len(sb))
    return len(sa & sb) / d if d else 0.0

def build_precomp(rec_map):
    out = {}
    for eid, item in rec_map.items():
        name, addr, ctry, nc, nph, nsk, ac = item[:7]
        out[eid] = Precomp(
            n_norm=normalize(name), n_core=nc, n_ph=nph, n_sk=nsk,
            a_norm=normalize(addr), a_core=ac, country=ctry,
            num_name=frozenset(num_tokens(name)),
            num_addr=frozenset(num_tokens(addr)),
            tok1=_first_tok(name), acronym=_acronym(nc),
            squashed_name=nc.replace(" ", ""),
            city=_city(ac), state=_state(ac),
            street_num=_leading_num(addr), postal=_postal(addr),
            street_name=_street_name(ac), n_tok_count=len(nc.split())
        )
    return out

def _bsim_batch(a_list, b_list, scorer, scale=1.0, workers=1):
    n = len(a_list)
    if n == 0:
        return np.empty(0, np.float32)

    out = np.zeros(n, dtype=np.float32)
    a_e = np.fromiter((not a for a in a_list), bool, n)
    b_e = np.fromiter((not b for b in b_list), bool, n)
    both_e = a_e & b_e
    active = ~(a_e | b_e)

    out[both_e] = 1.0
    if active.any():
        idx = np.where(active)[0]
        scores = process.cpdist(
            [a_list[i] for i in idx],
            [b_list[i] for i in idx],
            scorer=scorer,
            workers=workers,
        ).astype(np.float32, copy=False)
        if scale != 1.0:
            scores *= np.float32(scale)
        out[idx] = scores
    return out

def compute_features_batch(pairs, precomp):
    if not pairs:
        return np.empty((0, N_FEATURES), np.float32), []

    rows, ids = [], []
    for s1, cid in pairs:
        p1, p2 = precomp.get(s1), precomp.get(cid)
        if p1 is not None and p2 is not None:
            rows.append((p1, p2))
            ids.append((s1, cid))

    n = len(rows)
    if not n:
        return np.empty((0, N_FEATURES), np.float32), []

    p1s = [r[0] for r in rows]
    p2s = [r[1] for r in rows]
    workers = max(1, min(FEATURE_THREADS, 4))

    f_name_jw   = _bsim_batch([p.n_norm for p in p1s], [p.n_norm for p in p2s], JaroWinkler.similarity, workers=workers)
    f_name_jwc  = _bsim_batch([p.n_core for p in p1s], [p.n_core for p in p2s], JaroWinkler.similarity, workers=workers)
    f_name_ts   = _bsim_batch([p.n_norm for p in p1s], [p.n_norm for p in p2s], fuzz.token_sort_ratio, .01, workers)
    f_name_tset = _bsim_batch([p.n_norm for p in p1s], [p.n_norm for p in p2s], fuzz.token_set_ratio, .01, workers)
    f_name_pt   = _bsim_batch([p.n_norm for p in p1s], [p.n_norm for p in p2s], fuzz.partial_ratio, .01, workers)
    f_name_sk   = _bsim_batch([p.n_sk for p in p1s], [p.n_sk for p in p2s], JaroWinkler.similarity, workers=workers)
    f_name_phj  = _bsim_batch([p.n_ph for p in p1s], [p.n_ph for p in p2s], JaroWinkler.similarity, workers=workers)
    f_addr_jw   = _bsim_batch([p.a_norm for p in p1s], [p.a_norm for p in p2s], JaroWinkler.similarity, workers=workers)
    f_addr_jwc  = _bsim_batch([p.a_core for p in p1s], [p.a_core for p in p2s], JaroWinkler.similarity, workers=workers)
    f_addr_ts   = _bsim_batch([p.a_norm for p in p1s], [p.a_norm for p in p2s], fuzz.token_sort_ratio, .01, workers)
    f_addr_pt   = _bsim_batch([p.a_norm for p in p1s], [p.a_norm for p in p2s], fuzz.partial_ratio, .01, workers)

    l1 = np.fromiter((len(p.n_norm) for p in p1s), np.float32, n)
    l2 = np.fromiter((len(p.n_norm) for p in p2s), np.float32, n)
    mx, mn = np.maximum(l1, l2), np.minimum(l1, l2)
    f_len = np.where(mx > 0, mn / np.maximum(mx, 1e-6), 1.0)

    f_name_common = np.fromiter((_tok_ov(a.n_core, b.n_core) for a,b in rows), np.float32, n)
    f_addr_common = np.fromiter((_tok_ov(a.a_core, b.a_core) for a,b in rows), np.float32, n)
    f_addr_num = np.fromiter((_num_ov(a.num_addr, b.num_addr) for a,b in rows), np.float32, n)
    a1e = np.fromiter((not p.a_norm.strip() for p in p1s), bool, n)
    a2e = np.fromiter((not p.a_norm.strip() for p in p2s), bool, n)

    X = np.column_stack([
        f_name_jw, f_name_jwc, f_name_ts, f_name_tset, f_name_pt,
        f_name_sk,
        np.fromiter((1.0 if (a.n_ph and a.n_ph == b.n_ph) else 0.0 for a,b in rows), np.float32, n),
        f_name_phj, f_len, f_name_common,
        f_addr_jw, f_addr_jwc, f_addr_ts, f_addr_pt,
        f_addr_common, f_addr_num, (a1e & a2e).astype(np.float32),
        (a1e ^ a2e).astype(np.float32),
        np.fromiter((_num_ov(a.num_name,b.num_name) for a,b in rows), np.float32, n),
        np.fromiter((1.0 if (a.num_name|a.num_addr)&(b.num_name|b.num_addr) else 0.0 for a,b in rows), np.float32, n),
        (f_name_jw + f_addr_jw)/2,
        (f_name_sk > .85).astype(np.float32),
        np.fromiter((1.0 if a.country == b.country else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.city and a.city == b.city else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.city and b.city and a.city != b.city else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.state and b.state and a.state != b.state else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.num_addr and b.num_addr and not (a.num_addr & b.num_addr) else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.street_num and b.street_num and a.street_num == b.street_num else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.postal and b.postal and a.postal == b.postal else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.tok1 and a.tok1 == b.tok1 else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((min(a.n_tok_count,b.n_tok_count)/max(a.n_tok_count,b.n_tok_count,1) for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.acronym and b.acronym and (a.acronym == b.acronym or a.acronym == b.squashed_name or b.acronym == a.squashed_name) else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((_common_prefix(a.squashed_name,b.squashed_name) for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.state and b.state and a.state == b.state else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.street_name and b.street_name and a.street_name == b.street_name else 0.0 for a,b in rows), np.float32, n),
        np.fromiter((1.0 if a.postal and b.postal and a.postal[:3] == b.postal[:3] else 0.0 for a,b in rows), np.float32, n),
        ((f_name_ts >= .9) & (f_addr_ts >= .7)).astype(np.float32),
    ]).astype(np.float32, copy=False)

    assert X.shape == (n, N_FEATURES), X.shape
    return X, ids
