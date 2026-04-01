import json
import numpy as np
import pandas as pd
from collections import defaultdict
from .config import (
    RATINGS_FILE, MOVIE_INDEX_FILE,
    PRIORITY_LANGUAGES, PRIORITY_COUNTRIES, TOP_N, MIN_VOTE_COUNT, MIN_TMDB_SCORE,
)
from .tmdb import TMDBClient


def load_ratings() -> dict:
    if not RATINGS_FILE.exists():
        raise SystemExit("No ratings found. Run:  python recommend.py ingest")
    with open(RATINGS_FILE) as f:
        return json.load(f)


def _priority_boost(movie: dict) -> float:
    lang = movie.get('original_language', '')
    lb = PRIORITY_LANGUAGES.get(lang, 1.0)

    # Check production_countries if available, else use origin_country
    countries = []
    for field in ('production_countries', 'origin_country'):
        val = movie.get(field, [])
        if isinstance(val, list):
            for c in val:
                if isinstance(c, dict):
                    countries.append(c.get('iso_3166_1', ''))
                elif isinstance(c, str):
                    countries.append(c)

    cb = max((PRIORITY_COUNTRIES.get(c, 1.0) for c in countries), default=1.0)
    return max(lb, cb)


def recommend(top_n: int = TOP_N):
    """
    Hybrid recommendation: use TMDB's own recommendations/similar endpoints
    (trained on millions of users) and cross-reference with user ratings.

    For each rated film, pull TMDB recs + similar. Candidates that appear
    as recommendations for multiple highly-rated films score higher.
    """
    ratings = load_ratings()
    rated_ids = {int(k) for k in ratings}
    client = TMDBClient()

    # Weight: how much each rated film contributes to recommendations
    # Films rated above mean get positive weight; below get negative
    scores_list = list(ratings.values())
    mean_r = np.mean(scores_list)
    print(f"  {len(ratings)} rated films, mean rating: {mean_r:.1f}/10")

    # Collect candidates from TMDB recommendations
    # candidate_id -> {score, sources, movie_data}
    candidates = defaultdict(lambda: {'score': 0.0, 'sources': [], 'data': None})

    for tid_str, user_score in ratings.items():
        tid = int(tid_str)
        weight = (user_score - mean_r) / 5.0   # normalized: -1 to +0.4 for mean=8

        # Only pull recommendations for films rated >= 6
        if user_score < 6:
            continue

        rated_movie = client.get_movie(tid)
        rated_title = rated_movie.get('title', '?') if rated_movie else '?'

        # TMDB recommendations (collaborative filtering) + similar (content)
        recs = client.get_recommendations(tid, pages=2)
        sims = client.get_similar(tid, pages=1)

        for movie in recs:
            cid = movie['id']
            if cid in rated_ids:
                continue
            candidates[cid]['score'] += weight * 1.0   # full weight for recommendations
            candidates[cid]['sources'].append(rated_title)
            if candidates[cid]['data'] is None:
                candidates[cid]['data'] = movie

        for movie in sims:
            cid = movie['id']
            if cid in rated_ids:
                continue
            candidates[cid]['score'] += weight * 0.7   # reduced weight for similar
            if rated_title not in candidates[cid].get('sources', []):
                candidates[cid]['sources'].append(rated_title)
            if candidates[cid]['data'] is None:
                candidates[cid]['data'] = movie

    print(f"  {len(candidates)} candidate films found")

    # Also subtract weight for films similar to disliked ones
    for tid_str, user_score in ratings.items():
        if user_score >= 5:
            continue
        tid = int(tid_str)
        sims = client.get_similar(tid, pages=1)
        for movie in sims:
            cid = movie['id']
            if cid in candidates:
                penalty = (mean_r - user_score) / 10.0
                candidates[cid]['score'] -= penalty

    # Rank and filter
    results = []
    for cid, info in candidates.items():
        movie = info['data']
        if not movie:
            continue

        va = movie.get('vote_average', 0) or 0
        vc = movie.get('vote_count', 0) or 0
        if vc < MIN_VOTE_COUNT or va < MIN_TMDB_SCORE:
            continue

        raw_score = info['score']
        if raw_score <= 0:
            continue

        n_sources = len(info['sources'])
        boost = _priority_boost(movie)
        quality = va / 8.0

        # Final: accumulated cross-ref score * quality * language boost
        # Bonus for films recommended by multiple rated films
        final = raw_score * quality * boost * (1 + 0.15 * (n_sources - 1))

        results.append((final, n_sources, movie, info['sources']))

    results.sort(key=lambda r: r[0], reverse=True)
    return results[:top_n]


def print_recommendations(recs: list):
    n = len(recs)
    print(f"\n{'=' * 70}")
    print(f"  TOP {n} RECOMMENDATIONS")
    print(f"{'=' * 70}\n")

    for rank, (score, n_src, movie, sources) in enumerate(recs, 1):
        title = movie.get('title', '?')
        rd = movie.get('release_date', '') or ''
        year = rd[:4] if len(rd) >= 4 else ''
        yr = f" ({year})" if year else ''

        genres = [g['name'] if isinstance(g, dict) else g
                  for g in movie.get('genre_ids', movie.get('genres', []))]
        # genre_ids are ints from search results; map them
        GENRE_MAP = {
            28:'Action', 12:'Adventure', 16:'Animation', 35:'Comedy',
            80:'Crime', 99:'Documentary', 18:'Drama', 10751:'Family',
            14:'Fantasy', 36:'History', 27:'Horror', 10402:'Music',
            9648:'Mystery', 10749:'Romance', 878:'Sci-Fi', 10770:'TV Movie',
            53:'Thriller', 10752:'War', 37:'Western',
        }
        if genres and isinstance(genres[0], int):
            genres = [GENRE_MAP.get(g, '?') for g in genres]
        genre_str = ', '.join(genres[:3]) or 'N/A'

        lang = movie.get('original_language', '?')
        va = movie.get('vote_average', 0)

        src_str = ', '.join(sources[:3])
        if len(sources) > 3:
            src_str += f' +{len(sources)-3} more'

        print(f"  {rank:2d}. {title}{yr}")
        print(f"      {genre_str}  |  {lang}  |  TMDB {va:.1f}")
        print(f"      because you liked: {src_str}")
        print()
