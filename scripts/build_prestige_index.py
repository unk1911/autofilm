"""
Build data/prestige_index.json from canonical critical film lists.

Sources scraped:
  - Sight & Sound 2022 Greatest Films (BFI / Wikipedia)
  - They Shoot Pictures Don't They — Top 1000 (theyshootpictures.com)
  - Criterion Collection spine list (criterion.com)
  - Roger Ebert's Great Movies (rogerebert.com)

Each film is matched to a TMDB ID via title+year search.
The resulting JSON maps str(tmdb_id) → prestige_score (1.0 – 2.0).

Run once (takes ~5 min due to TMDB API rate limiting):
  python scripts/build_prestige_index.py
"""

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tmdb import TMDBClient

OUTPUT = Path('data/prestige_index.json')

# ---------------------------------------------------------------------------
# Scrapers — each returns list of (title, year_or_None, rank_or_None, source)
# ---------------------------------------------------------------------------

def _get(url: str, **kwargs) -> requests.Response:
    headers = {'User-Agent': 'autofilm/1.0 (film recommendation research)'}
    r = requests.get(url, headers=headers, timeout=15, **kwargs)
    r.raise_for_status()
    return r


def scrape_sight_and_sound() -> list[tuple]:
    """Sight & Sound 2022 Greatest Films poll — top 10 critics + top 10 directors from Wikipedia."""
    print("Scraping Sight & Sound 2022 ...")
    url = 'https://en.wikipedia.org/wiki/The_Sight_and_Sound_Greatest_Films_of_All_Time_2022'
    soup = BeautifulSoup(_get(url).text, 'html.parser')

    results = []
    seen = set()

    # The page has numbered <ol> or <li> lists under headings for critics/directors polls.
    # Each <li> contains a wikilink with the film title.
    for ol in soup.find_all('ol'):
        for rank, li in enumerate(ol.find_all('li'), start=1):
            text = li.get_text(strip=True)
            text = re.sub(r'\[.*?\]', '', text).strip()
            # Extract year from parentheses
            year = None
            m = re.search(r'\((\d{4})\)', text)
            if m:
                year = int(m.group(1))
                text = text[:m.start()].strip()
            # Strip vote counts like "215 votes, 13.1%"
            text = re.sub(r'\d+ votes.*', '', text).strip().rstrip(',').strip()
            # Try to get clean title from the first wikilink
            link = li.find('a')
            if link and link.get('title'):
                title = link['title']
                title = re.sub(r'\[.*?\]', '', title).strip()
            else:
                title = text
            if not title or title in seen:
                continue
            seen.add(title)
            results.append((title, year, rank, 'sight_and_sound'))

    print(f"  {len(results)} films")
    return results


def scrape_tspdt() -> list[tuple]:
    """They Shoot Pictures Don't They — top 1000."""
    print("Scraping TSPDT top 1000 ...")
    url = 'https://www.theyshootpictures.com/gf1000_all1000films_table.php'
    try:
        soup = BeautifulSoup(_get(url).text, 'html.parser')
    except Exception:
        # Fallback: try the alternate URL format
        url = 'https://www.theyshootpictures.com/gf1000.htm'
        soup = BeautifulSoup(_get(url).text, 'html.parser')

    results = []
    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 3:
            continue
        rank_text = cells[0].get_text(strip=True)
        try:
            rank = int(rank_text)
        except ValueError:
            continue
        title = cells[1].get_text(strip=True)
        title = re.sub(r'\[.*?\]', '', title).strip()
        year = None
        m = re.search(r'\((\d{4})\)', cells[2].get_text(strip=True))
        if m:
            year = int(m.group(1))
        if not year:
            m = re.search(r'\b(1[89]\d{2}|20[012]\d)\b', title)
            if m:
                year = int(m.group(1))
                title = title[:m.start()].strip()
        results.append((title, year, rank, 'tspdt'))

    print(f"  {len(results)} films")
    return results


def scrape_criterion() -> list[tuple]:
    """Criterion Collection — all spine entries."""
    print("Scraping Criterion Collection ...")
    url = 'https://www.criterion.com/shop/browse/list'
    soup = BeautifulSoup(_get(url).text, 'html.parser')

    results = []
    for item in soup.select('li.product_info, tr.gridFilm, .gridItem, li[class*="item"]'):
        title_tag = item.select_one('.title, .product-title, h3, h4, .gridTitle')
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        title = re.sub(r'\[.*?\]', '', title).strip()
        year = None
        year_tag = item.select_one('.year, .gridYear, .product-year')
        if year_tag:
            m = re.search(r'\b(1[89]\d{2}|20[012]\d)\b', year_tag.get_text())
            if m:
                year = int(m.group(1))
        if not year:
            m = re.search(r'\((\d{4})\)', title)
            if m:
                year = int(m.group(1))
                title = title[:m.start()].strip()
        if title:
            results.append((title, year, None, 'criterion'))

    # If the main selector didn't find anything, try a broader fallback
    if not results:
        for link in soup.select('a[href*="/spine/"]'):
            title = link.get_text(strip=True)
            title = re.sub(r'\[.*?\]', '', title).strip()
            if title:
                results.append((title, None, None, 'criterion'))

    print(f"  {len(results)} films")
    return results


def scrape_ebert() -> list[tuple]:
    """Roger Ebert's Great Movies list."""
    print("Scraping Roger Ebert Great Movies ...")
    url = 'https://www.rogerebert.com/great-movies'
    soup = BeautifulSoup(_get(url).text, 'html.parser')

    results = []
    for item in soup.select('article, .review-stack--movie, .movie-review'):
        title_tag = item.select_one('h2, h3, .title, .review-title')
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        title = re.sub(r'\[.*?\]', '', title).strip()
        year = None
        m = re.search(r'\((\d{4})\)', title)
        if m:
            year = int(m.group(1))
            title = title[:m.start()].strip()
        if title:
            results.append((title, year, None, 'ebert'))

    print(f"  {len(results)} films")
    return results


# ---------------------------------------------------------------------------
# TMDB matching
# ---------------------------------------------------------------------------

def match_to_tmdb(films: list[tuple], client: TMDBClient) -> dict[str, float]:
    """
    Match (title, year, rank, source) tuples to TMDB IDs.
    Returns {str(tmdb_id): prestige_score}.
    """
    scores: dict[str, dict] = {}  # tmdb_id_str -> {criterion, ss_rank, tspdt_rank, ebert}

    total = len(films)
    for i, (title, year, rank, source) in enumerate(films):
        if (i + 1) % 50 == 0:
            print(f"  matching {i+1}/{total} ...")

        # Search TMDB
        result = client.search(title, year=year)

        if not result and year:
            # Retry without year constraint
            result = client.search(title)

        if not result:
            continue

        tmdb_id = str(result.get('id', ''))
        if not tmdb_id:
            continue

        if tmdb_id not in scores:
            scores[tmdb_id] = {'criterion': False, 'ss_rank': None, 'tspdt_rank': None, 'ebert': False}

        if source == 'criterion':
            scores[tmdb_id]['criterion'] = True
        elif source == 'sight_and_sound' and rank is not None:
            existing = scores[tmdb_id]['ss_rank']
            scores[tmdb_id]['ss_rank'] = rank if existing is None else min(existing, rank)
        elif source == 'tspdt' and rank is not None:
            existing = scores[tmdb_id]['tspdt_rank']
            scores[tmdb_id]['tspdt_rank'] = rank if existing is None else min(existing, rank)
        elif source == 'ebert':
            scores[tmdb_id]['ebert'] = True

    # Compute final prestige score
    prestige_index: dict[str, float] = {}
    for tmdb_id, s in scores.items():
        score = 1.0
        if s['criterion']:
            score += 0.3
        if s['ss_rank'] is not None:
            score += 0.3 * (1 - s['ss_rank'] / 250)
        if s['tspdt_rank'] is not None:
            score += 0.2 * (1 - s['tspdt_rank'] / 1000)
        if s['ebert']:
            score += 0.1
        prestige_index[tmdb_id] = round(min(score, 2.0), 4)

    return prestige_index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_films = []

    for name, fn in [
        ('Sight & Sound', scrape_sight_and_sound),
        ('TSPDT',         scrape_tspdt),
        ('Criterion',     scrape_criterion),
        ('Ebert',         scrape_ebert),
    ]:
        try:
            all_films += fn()
        except Exception as e:
            print(f"  WARNING: {name} scraper failed ({e}) — skipping")

    print(f"\nTotal entries to match: {len(all_films)}")
    print("Matching to TMDB IDs (this will take a few minutes) ...")

    client = TMDBClient()
    prestige_index = match_to_tmdb(all_films, client)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(prestige_index, f, indent=2)

    print(f"\nSaved {len(prestige_index)} entries → {OUTPUT}")

    # Quick stats
    scores = list(prestige_index.values())
    above_15 = sum(1 for s in scores if s >= 1.5)
    above_18 = sum(1 for s in scores if s >= 1.8)
    print(f"Score distribution: {len(scores)} total, {above_15} with score ≥1.5, {above_18} with score ≥1.8")


if __name__ == '__main__':
    main()
