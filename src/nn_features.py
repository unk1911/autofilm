import numpy as np
import pandas as pd
from collections import Counter
from .config import DATA_DIR
from .tmdb import TMDBClient

FEATURES_FILE = DATA_DIR / 'nn_features.npz'
VOCAB_FILE = DATA_DIR / 'nn_vocab.npz'


def _decade(year):
    if not year or year < 1950:
        return 'pre1950'
    d = (year // 10) * 10
    key = f'{d}s'
    return key if key in ['pre1950', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'] else '2020s'


def build_vocab(movies_data: list):
    """Extract vocabulary from all movies."""
    all_genres = set()
    all_keywords = Counter()
    all_directors = Counter()
    all_countries = set()
    all_languages = set()

    for m in movies_data:
        for g in m.get('genres', []):
            all_genres.add(g['name'] if isinstance(g, dict) else g)
        for k in m.get('keywords', {}).get('keywords', []):
            all_keywords[k['name']] += 1
        for c in m.get('credits', {}).get('crew', []):
            if c.get('job') == 'Director':
                all_directors[c['name']] += 1
        for c in m.get('production_countries', []):
            all_countries.add(c.get('iso_3166_1', ''))
        all_languages.add(m.get('original_language', ''))

    # Top keywords/directors
    top_kw = [w for w, _ in all_keywords.most_common(500)]
    top_dir = [d for d, _ in all_directors.most_common(300)]

    vocab = {
        'genres': sorted(all_genres),
        'keywords': top_kw,
        'directors': top_dir,
        'countries': sorted(all_countries),
        'languages': sorted(all_languages),
        'decades': ['pre1950', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'],
    }
    return vocab


def featurize_movie(movie: dict, vocab: dict) -> np.ndarray:
    """Convert a single movie to a feature vector."""
    features = []

    # Genres (one-hot)
    genres = set(g['name'] if isinstance(g, dict) else g for g in movie.get('genres', []))
    for g in vocab['genres']:
        features.append(1.0 if g in genres else 0.0)

    # Top keywords (one-hot)
    keywords = set(k['name'] for k in movie.get('keywords', {}).get('keywords', []))
    for k in vocab['keywords']:
        features.append(1.0 if k in keywords else 0.0)

    # Top directors (one-hot)
    directors = set(c['name'] for c in movie.get('credits', {}).get('crew', []) if c.get('job') == 'Director')
    for d in vocab['directors']:
        features.append(1.0 if d in directors else 0.0)

    # Language (one-hot)
    lang = movie.get('original_language', '')
    for l in vocab['languages']:
        features.append(1.0 if l == lang else 0.0)

    # Countries (one-hot)
    countries = set(c.get('iso_3166_1', '') for c in movie.get('production_countries', []))
    for c in vocab['countries']:
        features.append(1.0 if c in countries else 0.0)

    # Decade (one-hot)
    rd = movie.get('release_date', '') or ''
    year = None
    if len(rd) >= 4:
        try:
            year = int(rd[:4])
        except ValueError:
            pass
    decade = _decade(year)
    for d in vocab['decades']:
        features.append(1.0 if d == decade else 0.0)

    # TMDB vote average (normalized 0-10 → 0-1)
    va = movie.get('vote_average', 0)
    features.append(min(va / 10.0, 1.0))

    # TMDB vote count (log scale)
    vc = movie.get('vote_count', 0)
    features.append(np.log1p(vc) / 10.0)

    # Runtime (minutes, normalized)
    runtime = movie.get('runtime', 0) or 90
    features.append(min(runtime / 300.0, 1.0))

    # Budget (log scale if present)
    budget = movie.get('budget', 0)
    features.append(np.log1p(budget) / 20.0 if budget > 0 else 0.0)

    return np.array(features, dtype=np.float32)


def build_training_data(ratings: dict):
    """Build feature matrix and target vector for training."""
    client = TMDBClient()

    # Load all cached movies
    movies_data = list(client.all_cached())
    vocab = build_vocab(movies_data)

    # Build ID → index mapping
    id_to_idx = {}
    features_list = []
    target_list = []

    for i, movie in enumerate(movies_data):
        id_to_idx[movie['id']] = i
        features_list.append(featurize_movie(movie, vocab))

    # Rated films (positive examples)
    rated_ids = set(int(k) for k in ratings.keys())
    train_indices = []
    train_targets = []

    for tid_str, score in ratings.items():
        tid = int(tid_str)
        if tid in id_to_idx:
            idx = id_to_idx[tid]
            train_indices.append(idx)
            train_targets.append(score / 10.0)  # normalize to 0-1

    n_positive = len(train_indices)

    # Negative sampling: random unseen films as implicit "average" scores.
    # This teaches the model what distinguishes your taste from generic films.
    rng = np.random.RandomState(42)
    unrated_indices = [i for i in range(len(movies_data)) if movies_data[i]['id'] not in rated_ids]
    n_negatives = min(len(unrated_indices), n_positive * 15)  # 15x negative ratio
    neg_sample = rng.choice(unrated_indices, size=n_negatives, replace=False)

    for idx in neg_sample:
        train_indices.append(idx)
        train_targets.append(0.45)  # implicit "meh" — below user's mean

    print(f"  {n_positive} rated + {n_negatives} negative samples = {len(train_indices)} training examples")

    X_train = np.array([features_list[i] for i in train_indices], dtype=np.float32)
    y_train = np.array(train_targets, dtype=np.float32)

    # Full feature matrix for inference
    X_all = np.array(features_list, dtype=np.float32)

    # Save
    np.savez_compressed(FEATURES_FILE, X_train=X_train, y_train=y_train, X_all=X_all)
    np.savez_compressed(
        VOCAB_FILE,
        genres=np.array(vocab['genres'], dtype=object),
        keywords=np.array(vocab['keywords'], dtype=object),
        directors=np.array(vocab['directors'], dtype=object),
        countries=np.array(vocab['countries'], dtype=object),
        languages=np.array(vocab['languages'], dtype=object),
        decades=np.array(vocab['decades'], dtype=object),
    )

    return X_train, y_train, X_all, vocab, id_to_idx


def load_training_data():
    """Load pre-built training data."""
    data = np.load(FEATURES_FILE)
    return data['X_train'], data['y_train'], data['X_all']


def get_movie_ids_all():
    """Get the full list of TMDB IDs in order."""
    client = TMDBClient()
    movies_data = list(client.all_cached())
    return np.array([m['id'] for m in movies_data], dtype=np.int64)
