import json
import numpy as np
import pandas as pd
from collections import Counter
from .config import (
    FEATURES_FILE, MOVIE_INDEX_FILE, DATA_DIR,
    PRIORITY_LANGUAGES, PRIORITY_COUNTRIES,
    MAX_KEYWORDS, MAX_DIRECTORS, MAX_ACTORS, TOP_CAST_PER_FILM,
    DECADES, LANGUAGES, COUNTRIES,
)
from .tmdb import TMDBClient

VOCAB_FILE = DATA_DIR / 'vocab.json'


def _decade(year):
    if not year or year < 1950:
        return 'pre1950'
    d = (year // 10) * 10
    key = f'{d}s'
    return key if key in DECADES else '2020s'


def _extract(movie: dict) -> dict:
    """Pull the fields we care about out of a raw TMDB response."""
    genres = [g['name'] for g in movie.get('genres', [])]

    lang = movie.get('original_language', 'other')
    if lang not in PRIORITY_LANGUAGES:
        lang = 'other'

    countries = [c['iso_3166_1'] for c in movie.get('production_countries', [])]
    country = next((c for c in countries if c in PRIORITY_COUNTRIES), 'other')

    year = None
    rd = movie.get('release_date', '') or ''
    if len(rd) >= 4:
        try:
            year = int(rd[:4])
        except ValueError:
            pass

    kw_list = movie.get('keywords', {}).get('keywords', [])
    keywords = [k['name'] for k in kw_list]

    crew = movie.get('credits', {}).get('crew', [])
    directors = [p['name'] for p in crew if p.get('job') == 'Director']

    cast = movie.get('credits', {}).get('cast', [])
    cast = sorted(cast, key=lambda x: x.get('order', 999))
    actors = [p['name'] for p in cast[:TOP_CAST_PER_FILM]]

    return {
        'tmdb_id':      movie['id'],
        'title':        movie.get('title', ''),
        'year':         year,
        'genres':       genres,
        'language':     lang,
        'country':      country,
        'decade':       _decade(year),
        'keywords':     keywords,
        'directors':    directors,
        'actors':       actors,
        'vote_average': movie.get('vote_average', 0),
        'vote_count':   movie.get('vote_count', 0),
    }


def build_features():
    """Scan all cached TMDB data, build vocab + feature matrix, save to disk."""
    client = TMDBClient()

    print("Extracting fields from cached movies ...")
    records = []
    for movie in client.all_cached():
        try:
            records.append(_extract(movie))
        except Exception:
            continue

    print(f"  {len(records):,} movies loaded")

    # --- vocabulary (most-frequent keywords, directors, actors) ---
    kw_cnt   = Counter()
    dir_cnt  = Counter()
    act_cnt  = Counter()
    genre_set = set()

    for r in records:
        kw_cnt.update(r['keywords'])
        dir_cnt.update(r['directors'])
        act_cnt.update(r['actors'])
        genre_set.update(r['genres'])

    top_kw   = [w for w, _ in kw_cnt.most_common(MAX_KEYWORDS)]
    top_dir  = [d for d, _ in dir_cnt.most_common(MAX_DIRECTORS)]
    top_act  = [a for a, _ in act_cnt.most_common(MAX_ACTORS)]
    all_genres = sorted(genre_set)

    vocab = {
        'genres': all_genres, 'keywords': top_kw,
        'directors': top_dir, 'actors': top_act,
        'languages': LANGUAGES, 'countries': COUNTRIES, 'decades': DECADES,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(VOCAB_FILE, 'w') as f:
        json.dump(vocab, f)

    # --- fast lookups ---
    kw_idx   = {w: i for i, w in enumerate(top_kw)}
    dir_idx  = {d: i for i, d in enumerate(top_dir)}
    act_idx  = {a: i for i, a in enumerate(top_act)}
    genre_idx = {g: i for i, g in enumerate(all_genres)}
    lang_idx  = {l: i for i, l in enumerate(LANGUAGES)}
    cty_idx   = {c: i for i, c in enumerate(COUNTRIES)}
    dec_idx   = {d: i for i, d in enumerate(DECADES)}

    n_feat = (len(all_genres) + MAX_KEYWORDS + MAX_DIRECTORS
              + MAX_ACTORS + len(LANGUAGES) + len(COUNTRIES) + len(DECADES))
    print(f"Feature vector dimension: {n_feat}")

    # --- encode every movie ---
    ids     = np.empty(len(records), dtype=np.int64)
    X       = np.zeros((len(records), n_feat), dtype=np.float32)
    meta_rows = []

    # offsets into the flat vector
    o1 = 0;                     o2 = len(all_genres)
    o3 = o2;                    o4 = o3 + MAX_KEYWORDS
    o5 = o4;                    o6 = o5 + MAX_DIRECTORS
    o7 = o6;                    o8 = o7 + MAX_ACTORS
    o9 = o8;                    o10 = o9 + len(LANGUAGES)
    o11 = o10;                  o12 = o11 + len(COUNTRIES)
    o13 = o12  # decades start

    # feature-group weights (tuned for auteur/art-house taste profiles)
    W_GENRE = 1.5;  W_KW = 2.0;  W_DIR = 2.0
    W_ACT   = 0.5;  W_LANG = 0.8; W_CTY = 0.5; W_DEC = 0.3

    for i, r in enumerate(records):
        ids[i] = r['tmdb_id']

        for g in r['genres']:
            j = genre_idx.get(g)
            if j is not None:
                X[i, o1 + j] = W_GENRE

        for w in r['keywords']:
            j = kw_idx.get(w)
            if j is not None:
                X[i, o3 + j] = W_KW

        for d in r['directors']:
            j = dir_idx.get(d)
            if j is not None:
                X[i, o5 + j] = W_DIR

        for a in r['actors']:
            j = act_idx.get(a)
            if j is not None:
                X[i, o7 + j] = W_ACT

        j = lang_idx.get(r['language'], lang_idx['other'])
        X[i, o9 + j] = W_LANG

        j = cty_idx.get(r['country'], cty_idx['other'])
        X[i, o11 + j] = W_CTY

        j = dec_idx.get(r['decade'], 0)
        X[i, o13 + j] = W_DEC

        meta_rows.append({
            'tmdb_id':      r['tmdb_id'],
            'title':        r['title'],
            'year':         r['year'],
            'language':     r['language'],
            'country':      r['country'],
            'genres':       '|'.join(r['genres']),
            'vote_average': r['vote_average'],
            'vote_count':   r['vote_count'],
        })

    # IDF weighting: rare features (specific directors, niche keywords) matter
    # more than ubiquitous ones ("Drama", "based on novel").
    # Capped to prevent ultra-rare features from dominating.
    doc_freq = np.sum(X > 0, axis=0).astype(np.float32)
    doc_freq[doc_freq == 0] = 1.0
    idf = np.log(len(records) / doc_freq)
    idf = np.clip(idf, 0.5, 4.0)
    X *= idf[np.newaxis, :]
    print(f"  IDF applied (capped 0.5–4.0)")

    # L2-normalise each row (so dot product == cosine similarity)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms

    np.savez_compressed(FEATURES_FILE, X=X, ids=ids)
    pd.DataFrame(meta_rows).to_csv(MOVIE_INDEX_FILE, index=False)

    print(f"Feature matrix saved  → {FEATURES_FILE}  ({X.shape})")
    print(f"Movie index saved     → {MOVIE_INDEX_FILE}")
    return X, ids


def load_features():
    """Load pre-built feature matrix and ID array."""
    data = np.load(FEATURES_FILE)
    return data['X'], data['ids']
