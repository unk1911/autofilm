import io
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
from .config import MOVIELENS_URL, ML_DIR
from .tmdb import TMDBClient


def download_movielens():
    """Download MovieLens ml-latest and extract movies.csv + links.csv."""
    ML_DIR.mkdir(parents=True, exist_ok=True)
    movies_csv = ML_DIR / 'movies.csv'
    links_csv  = ML_DIR / 'links.csv'

    if movies_csv.exists() and links_csv.exists():
        print("MovieLens data already present — skipping download.")
        return

    print("Downloading MovieLens ml-latest (~330 MB) ...")
    resp = requests.get(MOVIELENS_URL, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get('content-length', 0))
    buf = io.BytesIO()

    with tqdm(total=total, unit='B', unit_scale=True, desc='Downloading') as pbar:
        for chunk in resp.iter_content(chunk_size=256 * 1024):
            buf.write(chunk)
            pbar.update(len(chunk))

    buf.seek(0)
    print("Extracting movies.csv and links.csv ...")

    with zipfile.ZipFile(buf) as zf:
        for name in zf.namelist():
            basename = name.split('/')[-1]
            if basename in ('movies.csv', 'links.csv'):
                with zf.open(name) as src, open(ML_DIR / basename, 'wb') as dst:
                    dst.write(src.read())

    print("MovieLens data ready.")


def load_catalog() -> pd.DataFrame:
    """Return DataFrame with columns: movieId, title, genres, tmdbId."""
    movies = pd.read_csv(ML_DIR / 'movies.csv')
    links  = pd.read_csv(ML_DIR / 'links.csv')

    links['tmdbId'] = pd.to_numeric(links['tmdbId'], errors='coerce')
    links = links.dropna(subset=['tmdbId'])
    links['tmdbId'] = links['tmdbId'].astype(int)

    return movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='inner')


def fetch_all_metadata(limit: int = None):
    """
    Fetch TMDB metadata for every movie in the catalog.
    Already-cached movies are skipped, so this is safely resumable.
    """
    client  = TMDBClient()
    catalog = load_catalog()

    if limit:
        catalog = catalog.head(limit)

    tmdb_ids   = catalog['tmdbId'].tolist()
    to_fetch   = [tid for tid in tmdb_ids if not client.is_cached(tid)]
    already    = len(tmdb_ids) - len(to_fetch)

    print(f"Catalog: {len(tmdb_ids):,} movies | cached: {already:,} | to fetch: {len(to_fetch):,}")

    if not to_fetch:
        print("All metadata already cached.")
        return

    print("Fetching from TMDB API (this takes a while — it's a one-time step) ...")
    with tqdm(total=len(to_fetch), desc='TMDB fetch', unit='film') as pbar:
        for tid in to_fetch:
            client.get_movie(tid)
            pbar.update(1)

    print(f"Done. Total cached: {client.cached_count():,}")
