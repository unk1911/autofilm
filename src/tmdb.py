import os
import json
import time
import sqlite3
import requests
from pathlib import Path
from .config import TMDB_CACHE_DB

BASE_URL = 'https://api.themoviedb.org/3'


def get_api_key():
    key = os.environ.get('TMDB_API_KEY', '').strip()
    if not key:
        raise SystemExit(
            "\nTMDB_API_KEY not set.\n"
            "  1. Go to https://www.themoviedb.org/settings/api (free account)\n"
            "  2. Copy your API key (v3 auth)\n"
            "  3. Run:  export TMDB_API_KEY=your_key_here\n"
        )
    return key


class TMDBClient:
    def __init__(self):
        self.api_key = get_api_key()
        self.session = requests.Session()
        self.session.params = {'api_key': self.api_key, 'language': 'en-US'}
        self._init_db()

    def _init_db(self):
        TMDB_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(TMDB_CACHE_DB))
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                tmdb_id   INTEGER PRIMARY KEY,
                data      TEXT    NOT NULL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS failed (
                tmdb_id INTEGER PRIMARY KEY,
                reason  TEXT
            )
        ''')
        self.conn.commit()

    def is_cached(self, tmdb_id: int) -> bool:
        hit = self.conn.execute(
            'SELECT 1 FROM movies WHERE tmdb_id=?', (tmdb_id,)
        ).fetchone()
        if hit:
            return True
        return bool(self.conn.execute(
            'SELECT 1 FROM failed WHERE tmdb_id=?', (tmdb_id,)
        ).fetchone())

    def get_movie(self, tmdb_id: int, force: bool = False):
        """Return full movie record (details + keywords + credits), cached in SQLite."""
        if not force:
            row = self.conn.execute(
                'SELECT data FROM movies WHERE tmdb_id=?', (tmdb_id,)
            ).fetchone()
            if row:
                return json.loads(row[0])

        try:
            resp = self.session.get(
                f'{BASE_URL}/movie/{tmdb_id}',
                params={'append_to_response': 'keywords,credits'},
                timeout=15,
            )
            if resp.status_code == 404:
                self.conn.execute(
                    'INSERT OR REPLACE INTO failed VALUES (?,?)', (tmdb_id, 'not_found')
                )
                self.conn.commit()
                return None
            resp.raise_for_status()
            data = resp.json()
            self.conn.execute(
                'INSERT OR REPLACE INTO movies (tmdb_id, data) VALUES (?,?)',
                (tmdb_id, json.dumps(data)),
            )
            self.conn.commit()
            time.sleep(0.025)  # stay under TMDB's 50 req/sec limit
            return data
        except requests.RequestException as e:
            self.conn.execute(
                'INSERT OR REPLACE INTO failed VALUES (?,?)', (tmdb_id, str(e)[:200])
            )
            self.conn.commit()
            return None

    def search(self, title: str, year: str = None):
        """Search TMDB by title (+ optional year). Returns the best match or None."""
        params = {'query': title}
        if year:
            params['primary_release_year'] = str(year)
        try:
            resp = self.session.get(f'{BASE_URL}/search/movie', params=params, timeout=15)
            resp.raise_for_status()
            results = resp.json().get('results', [])
            if not results:
                return None
            results.sort(key=lambda r: r.get('vote_count', 0), reverse=True)
            return results[0]
        except requests.RequestException:
            return None

    def get_recommendations(self, tmdb_id: int, pages: int = 2) -> list[dict]:
        """Get TMDB's own recommendations for a movie (collaborative filtering)."""
        all_results = []
        for page in range(1, pages + 1):
            try:
                resp = self.session.get(
                    f'{BASE_URL}/movie/{tmdb_id}/recommendations',
                    params={'page': page},
                    timeout=15,
                )
                if resp.status_code != 200:
                    break
                results = resp.json().get('results', [])
                if not results:
                    break
                all_results.extend(results)
                time.sleep(0.025)
            except requests.RequestException:
                break
        return all_results

    def get_similar(self, tmdb_id: int, pages: int = 1) -> list[dict]:
        """Get TMDB's similar movies (genre/keyword based)."""
        all_results = []
        for page in range(1, pages + 1):
            try:
                resp = self.session.get(
                    f'{BASE_URL}/movie/{tmdb_id}/similar',
                    params={'page': page},
                    timeout=15,
                )
                if resp.status_code != 200:
                    break
                results = resp.json().get('results', [])
                if not results:
                    break
                all_results.extend(results)
                time.sleep(0.025)
            except requests.RequestException:
                break
        return all_results

    def cached_count(self) -> int:
        return self.conn.execute('SELECT COUNT(*) FROM movies').fetchone()[0]

    def all_cached(self):
        """Yield all cached movie dicts (generator to avoid loading all into RAM)."""
        cur = self.conn.execute('SELECT data FROM movies')
        for (raw,) in cur:
            yield json.loads(raw)
