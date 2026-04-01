from pathlib import Path

LETTERBOXD_USERNAME = 'unk1911'

DATA_DIR = Path('data')
TMDB_CACHE_DB  = DATA_DIR / 'tmdb_cache.db'
RATINGS_FILE   = DATA_DIR / 'ratings.json'
ML_DIR         = DATA_DIR / 'movielens'
FEATURES_FILE  = DATA_DIR / 'features.npz'
MOVIE_INDEX_FILE = DATA_DIR / 'movie_index.csv'

MOVIELENS_URL = 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'

TOP_N = 20
MIN_VOTE_COUNT = 300   # filter out fan-inflated scores with few votes
MIN_TMDB_SCORE = 7.0   # quality floor

# Language priority boosts (ISO 639-1)
PRIORITY_LANGUAGES = {
    'en': 1.25,  # English
    'ru': 1.20,  # Russian
    'hr': 1.20,  # Croatian
    'fr': 1.15,  # French
    'ko': 1.20,  # South Korean
    'es': 1.15,  # Spanish
    'fi': 1.20,  # Finnish
    'sr': 1.20,  # Serbian
    'ja': 1.15,  # Japanese
    'ro': 1.15,  # Romanian
}

# Country priority boosts (ISO 3166-1 alpha-2)
PRIORITY_COUNTRIES = {
    'GB': 1.20,  # UK
    'US': 1.15,  # American
    'RU': 1.20,  # Russian
    'HR': 1.20,  # Croatian
    'FR': 1.15,  # French
    'KR': 1.20,  # South Korean
    'ES': 1.15,  # Spanish
    'AR': 1.20,  # Argentinian
    'FI': 1.20,  # Finnish
    'RS': 1.20,  # Serbian
    'JP': 1.15,  # Japanese
    'RO': 1.15,  # Romanian
}

# Feature vector dimensions
MAX_KEYWORDS  = 2000
MAX_DIRECTORS = 500
MAX_ACTORS    = 500
TOP_CAST_PER_FILM = 3

DECADES = ['pre1950', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
LANGUAGES = list(PRIORITY_LANGUAGES.keys()) + ['other']
COUNTRIES = list(PRIORITY_COUNTRIES.keys()) + ['other']
