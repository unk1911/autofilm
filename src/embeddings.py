import json
import textwrap
import numpy as np
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from .config import DATA_DIR, PRIORITY_LANGUAGES, PRIORITY_COUNTRIES, MIN_VOTE_COUNT, MIN_TMDB_SCORE
from .tmdb import TMDBClient
from .prestige import prestige_score

EMBEDDINGS_FILE = DATA_DIR / 'embeddings.npz'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

_LANG_NAMES = {
    'en': 'English-language', 'fr': 'French', 'ko': 'Korean',
    'ja': 'Japanese', 'ru': 'Russian', 'es': 'Spanish',
    'hr': 'Croatian', 'fi': 'Finnish', 'sr': 'Serbian',
    'ro': 'Romanian', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'zh': 'Chinese', 'hi': 'Hindi',
}


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


def recommend(ratings: dict, top_n: int = 20, nn_predictions: 'np.ndarray | None' = None, temperature: float = 0.0):
    """
    Recommend using semantic embeddings + anti-popularity bias.

    Taste profile = weighted mean of liked embeddings - weighted mean of disliked.
    Score = cosine similarity to taste profile * anti-popularity bias.

    If nn_predictions is provided (array aligned to all_cached() order, 0-10 scale),
    it is blended in as a ±20% nudge on top of the embedding-based score.
    """
    embeddings, ids = load_embeddings()
    if embeddings.ndim < 2 or embeddings.shape[0] == 0:
        raise RuntimeError(
            "Embeddings file is empty — run 'python recommend.py train' first"
        )
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

    # Build director affinity table from loved films (score >= 9)
    director_counts: Counter = Counter()
    movie_by_id = {m['id']: m for m in all_movies}
    for tid_str, score in ratings.items():
        if score >= 9:
            m = movie_by_id.get(int(tid_str))
            if m:
                for c in m.get('credits', {}).get('crew', []):
                    if c.get('job') == 'Director':
                        director_counts[c['name']] += 1

    def _director_boost(name: str) -> float:
        count = director_counts.get(name, 0)
        return 1.0 + 0.3 * min(count, 4)  # caps at 2.2x for 4+ loved films

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

        # Extract metadata for display
        crew = movie.get('credits', {}).get('crew', [])
        director = next((c['name'] for c in crew if c.get('job') == 'Director'), '?')
        rd = movie.get('release_date', '') or ''
        year = rd[:4] if len(rd) >= 4 else ''

        # NN blend: ±20% nudge from structured feature model (if available)
        nn_blend = 1.0
        if nn_predictions is not None:
            nn_blend = 0.8 + 0.4 * (float(nn_predictions[i]) / 10.0)

        # Similarity is king — square it so taste match dominates
        dir_boost = _director_boost(director)
        prest = prestige_score(tid)
        final = (sim ** 2) * quality * obscurity_boost * locale_boost * dir_boost * prest * nn_blend

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
            '_quality': quality,
            '_obscurity_boost': obscurity_boost,
            '_locale_boost': locale_boost,
            '_director_boost': dir_boost,
            '_prestige': prest,
            '_nn_blend': nn_blend,
        })

    # Apply temperature-based noise for result diversity across runs
    if temperature < 0:
        print("  Warning: temperature < 0 clamped to 0")
        temperature = 0.0
    if temperature > 1.0:
        print("  Warning: temperature > 1.0 may produce very unstable rankings")
    if temperature > 0:
        rng = np.random.default_rng()
        noise = rng.standard_normal(len(results))
        for idx, r in enumerate(results):
            r['score'] *= float(np.exp(temperature * noise[idx]))

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


def _build_explanation(r: dict) -> str:
    """Build a 2-3 sentence explanation of why this film was recommended."""
    parts = []

    # Taste similarity — always mentioned, primary signal
    sim = r['similarity']
    if sim >= 0.65:
        parts.append(f"Strong alignment with your taste profile (match {sim:.0%})")
    elif sim >= 0.50:
        parts.append(f"Good alignment with your taste profile (match {sim:.0%})")
    else:
        parts.append(f"Moderate thematic overlap with your preferences (match {sim:.0%})")

    # Director affinity
    db = r.get('_director_boost', 1.0)
    if db >= 1.6:
        parts.append(f"you've loved several films by {r['director']}")
    elif db >= 1.3:
        parts.append(f"you've enjoyed other work by {r['director']}")

    # Obscurity / hidden gem
    ob = r.get('_obscurity_boost', 1.0)
    vc = r.get('vote_count', 0)
    if ob >= 2.0:
        parts.append("this is a hidden gem with very few ratings")
    elif ob >= 1.5 and vc < 2000:
        parts.append("a lesser-known title that fits your taste")

    # Quality / critical acclaim
    va = r.get('tmdb_score', 0)
    if va >= 8.0:
        parts.append(f"highly rated at {va:.1f}/10 on TMDB")
    elif va >= 7.5:
        parts.append(f"well-reviewed ({va:.1f}/10)")

    # Prestige
    p = r.get('_prestige', 1.0)
    if p >= 1.15:
        parts.append("recognized across major film awards and curated lists")
    elif p >= 1.05:
        parts.append("noted in critical circles")

    # Locale boost
    lb = r.get('_locale_boost', 1.0)
    if lb > 1.0:
        lang = r.get('language', '')
        lang_name = _LANG_NAMES.get(lang, lang)
        parts.append(f"matches your preference for {lang_name} cinema")

    # Assemble into prose
    if len(parts) <= 1:
        return parts[0] + '.' if parts else ''

    lead = parts[0]
    details = parts[1:]
    if len(details) == 1:
        return f"{lead} -- {details[0]}."
    else:
        body = ', '.join(details[:-1]) + f', and {details[-1]}'
        return f"{lead} -- {body}."


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

        explanation = _build_explanation(r)
        if explanation:
            wrapped = textwrap.fill(explanation, width=64,
                                    initial_indent='      ',
                                    subsequent_indent='      ')
            print(wrapped)

        print()
