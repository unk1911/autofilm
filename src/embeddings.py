import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from .config import DATA_DIR, PRIORITY_LANGUAGES, PRIORITY_COUNTRIES, MIN_VOTE_COUNT, MIN_TMDB_SCORE
from .tmdb import TMDBClient

EMBEDDINGS_FILE = DATA_DIR / 'embeddings.npz'
MODEL_NAME = 'all-MiniLM-L6-v2'


def _movie_text(movie: dict) -> str:
    """Build a rich text description of a movie for embedding."""
    parts = []

    title = movie.get('title', '')
    parts.append(title)

    # Genres
    genres = [g['name'] for g in movie.get('genres', [])]
    if genres:
        parts.append(f"Genres: {', '.join(genres)}.")

    # Director
    crew = movie.get('credits', {}).get('crew', [])
    directors = [c['name'] for c in crew if c.get('job') == 'Director']
    if directors:
        parts.append(f"Directed by {', '.join(directors)}.")

    # Overview (the plot description — this is the richest signal)
    overview = movie.get('overview', '')
    if overview:
        parts.append(overview)

    # Keywords (often the most specific/distinctive tags)
    keywords = [k['name'] for k in movie.get('keywords', {}).get('keywords', [])]
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}.")

    # Country + language
    countries = [c.get('iso_3166_1', '') for c in movie.get('production_countries', [])]
    lang = movie.get('original_language', '')
    if countries:
        parts.append(f"Country: {', '.join(countries)}. Language: {lang}.")

    # Year
    rd = movie.get('release_date', '') or ''
    if len(rd) >= 4:
        parts.append(f"Year: {rd[:4]}.")

    return ' '.join(parts)


def build_embeddings():
    """Embed all cached movies using a sentence transformer."""
    client = TMDBClient()
    movies = list(client.all_cached())

    print(f"Building text descriptions for {len(movies):,} films ...")
    texts = [_movie_text(m) for m in movies]
    ids = np.array([m['id'] for m in movies], dtype=np.int64)

    print(f"Embedding with {MODEL_NAME} (GPU) ...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=256,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(np.float32)
    np.savez_compressed(EMBEDDINGS_FILE, embeddings=embeddings, ids=ids)
    print(f"Embeddings saved → {EMBEDDINGS_FILE}  shape={embeddings.shape}")
    return embeddings, ids


def load_embeddings():
    data = np.load(EMBEDDINGS_FILE)
    return data['embeddings'], data['ids']


def recommend(ratings: dict, top_n: int = 20):
    """
    Recommend using semantic embeddings + anti-popularity bias.

    Taste profile = weighted mean of liked embeddings - weighted mean of disliked.
    Score = cosine similarity to taste profile * anti-popularity bias.
    """
    embeddings, ids = load_embeddings()
    id_to_idx = {int(mid): i for i, mid in enumerate(ids)}

    client = TMDBClient()
    all_movies = list(client.all_cached())
    rated_ids = {int(k) for k in ratings}

    # Build taste profile using clear signal bands:
    #   9-10 = strong positive (films you love)
    #   6-8  = neutral (ignored — don't dilute the signal)
    #   1-5  = negative (films you dislike)
    scores_list = list(ratings.values())
    mean_r = np.mean(scores_list)
    n_pos = sum(1 for s in scores_list if s >= 9)
    n_neg = sum(1 for s in scores_list if s <= 5)
    print(f"  {len(ratings)} rated films, mean: {mean_r:.1f}/10")
    print(f"  Profile: {n_pos} loved (9-10), {n_neg} disliked (1-5), {len(ratings)-n_pos-n_neg} neutral (ignored)")

    pos_vec = np.zeros(embeddings.shape[1], dtype=np.float64)
    neg_vec = np.zeros(embeddings.shape[1], dtype=np.float64)
    pos_w = 0.0
    neg_w = 0.0
    missing = 0

    for tid_str, score in ratings.items():
        idx = id_to_idx.get(int(tid_str))
        if idx is None:
            missing += 1
            continue

        if score >= 9:
            # Weight 10s stronger than 9s
            w = 2.0 if score == 10 else 1.0
            pos_vec += w * embeddings[idx].astype(np.float64)
            pos_w += w
        elif score <= 5:
            w = (6 - score)  # 5→1, 4→2, 3→3, etc.
            neg_vec += w * embeddings[idx].astype(np.float64)
            neg_w += w
        # 6-8: neutral, skip

    if missing:
        print(f"  {missing} film(s) not in embeddings (skipped)")

    pos_vec /= max(pos_w, 1e-8)
    if neg_w > 0:
        neg_vec /= neg_w

    profile = pos_vec - 0.8 * neg_vec
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile /= norm
    profile = profile.astype(np.float32)

    # Cosine similarity (embeddings are already L2-normalized)
    similarities = embeddings @ profile

    # Score all films
    results = []
    for i, movie in enumerate(all_movies):
        tid = movie['id']
        if tid in rated_ids:
            continue

        va = movie.get('vote_average', 0) or 0
        vc = movie.get('vote_count', 0) or 0
        if vc < MIN_VOTE_COUNT or va < MIN_TMDB_SCORE:
            continue

        # Filter out shorts (< 60 min) unless anime
        runtime = movie.get('runtime', 0) or 0
        genres = [g['name'] for g in movie.get('genres', [])]
        is_anime = 'Animation' in genres and movie.get('original_language') == 'ja'
        if 0 < runtime < 60 and not is_anime:
            continue

        # Skip family/kids content (unless anime)
        if 'Family' in genres and not is_anime:
            continue

        # Skip TV movies and direct-to-video (low production value noise)
        if 'TV Movie' in genres:
            continue

        # Skip romance-dominated films (not the user's taste at all)
        if genres and genres[0] == 'Romance':
            continue

        sim = float(similarities[i])
        if sim < 0.25:
            continue

        # Anti-popularity bias: strongly boost obscure films, dampen blockbusters.
        popularity_penalty = 1.0 / (1.0 + np.log1p(vc) / 6.0)
        obscurity_boost = 1.0 + 1.8 * popularity_penalty

        # Blockbuster penalty: aggressively dampen films with huge vote counts
        if vc > 5000:
            obscurity_boost *= 0.7
        if vc > 15000:
            obscurity_boost *= 0.6

        # Quality factor — reward critically acclaimed films
        quality = (va / 10.0) ** 1.5

        # Language/country boost
        lang = movie.get('original_language', '')
        lb = PRIORITY_LANGUAGES.get(lang, 1.0)
        countries = [c.get('iso_3166_1', '') for c in movie.get('production_countries', [])]
        cb = max((PRIORITY_COUNTRIES.get(c, 1.0) for c in countries), default=1.0)
        locale_boost = max(lb, cb)

        # Similarity is king — square it so taste match dominates
        final = (sim ** 2) * quality * obscurity_boost * locale_boost

        # Extract metadata for display
        crew = movie.get('credits', {}).get('crew', [])
        director = next((c['name'] for c in crew if c.get('job') == 'Director'), '?')
        rd = movie.get('release_date', '') or ''
        year = rd[:4] if len(rd) >= 4 else ''

        results.append({
            'score': final,
            'similarity': sim,
            'title': movie.get('title', '?'),
            'year': year,
            'genres': genres,
            'director': director,
            'language': lang,
            'tmdb_score': va,
            'vote_count': vc,
        })

    results.sort(key=lambda r: r['score'], reverse=True)

    # Genre diversity: don't let one genre dominate the list.
    # Cap any single genre at ~40% of results.
    max_per_genre = max(top_n * 2 // 5, 4)
    genre_counts = {}
    diverse = []
    for r in results:
        primary = r['genres'][0] if r['genres'] else 'Unknown'
        genre_counts[primary] = genre_counts.get(primary, 0) + 1
        if genre_counts[primary] <= max_per_genre:
            diverse.append(r)
        if len(diverse) >= top_n:
            break

    return diverse


def print_recommendations(recs: list):
    n = len(recs)
    print(f"\n{'=' * 70}")
    print(f"  TOP {n} RECOMMENDATIONS")
    print(f"{'=' * 70}\n")

    for rank, r in enumerate(recs, 1):
        yr = f" ({r['year']})" if r['year'] else ''
        genre_str = ', '.join(r['genres'][:3]) or 'N/A'

        print(f"  {rank:2d}. {r['title']}{yr}")
        print(f"      {genre_str}  |  {r['language']}  |  TMDB {r['tmdb_score']:.1f}")
        print(f"      dir: {r['director']}  |  match: {r['similarity']:.2f}")
        print()
