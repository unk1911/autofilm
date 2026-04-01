import re
import csv
import xml.etree.ElementTree as ET
import requests
from pathlib import Path
from .config import LETTERBOXD_USERNAME, DATA_DIR

BASE = 'https://letterboxd.com'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
}
NS = {
    'letterboxd': 'https://letterboxd.com',
    'tmdb':       'https://themoviedb.org',
}


def scrape_ratings(username: str = LETTERBOXD_USERNAME) -> list[dict]:
    """
    Pull rated films from public Letterboxd RSS feed.
    Returns recent diary entries with ratings (typically ~16-50).
    Use with merge mode in ingest so new ratings accumulate over time.
    """
    url = f'{BASE}/{username}/rss/'
    print(f"Fetching Letterboxd RSS for @{username} ...")

    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    rated = []

    for item in root.iter('item'):
        rating_el = item.find('letterboxd:memberRating', NS)
        if rating_el is None or not rating_el.text:
            continue

        score = round(float(rating_el.text) * 2)  # 0.5–5.0 stars → 1–10

        title_el = item.find('letterboxd:filmTitle', NS)
        year_el  = item.find('letterboxd:filmYear', NS)
        tmdb_el  = item.find('tmdb:movieId', NS)

        title   = title_el.text.strip() if title_el is not None and title_el.text else ''
        year    = year_el.text.strip()  if year_el  is not None and year_el.text  else None
        tmdb_id = int(tmdb_el.text)     if tmdb_el  is not None and tmdb_el.text  else None

        if title:
            rated.append({
                'title': title, 'year': year, 'slug': '',
                'rating': score, 'tmdb_id': tmdb_id,
            })

    print(f"RSS returned {len(rated)} rated film(s)")
    print("  (new ratings merge with existing — nothing is lost)")
    return rated


def read_csv_export(csv_path: Path = None) -> list[dict]:
    """
    Read ratings from the official Letterboxd CSV export.
    Letterboxd → Settings → Import & Export → Export Your Data
    Place ratings.csv in data/letterboxd/
    """
    if csv_path is None:
        csv_path = DATA_DIR / 'letterboxd' / 'ratings.csv'

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}\n"
            "To export: Letterboxd → Settings → Import & Export → Export Your Data\n"
            "Then put ratings.csv in data/letterboxd/"
        )

    rated = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rating_str = row.get('Rating', '').strip()
            if not rating_str:
                continue
            try:
                score = round(float(rating_str) * 2)
            except ValueError:
                continue

            rated.append({
                'title':   row.get('Name', '').strip(),
                'year':    row.get('Year', '').strip() or None,
                'slug':    '',
                'rating':  score,
                'tmdb_id': None,
            })

    print(f"Loaded {len(rated)} ratings from CSV")
    return rated
