#!/usr/bin/env python3
"""
autofilm — personal movie recommendation engine

Usage:
  python recommend.py setup   [--limit N]   Download catalog + fetch TMDB metadata
  python recommend.py ingest  [--csv]       Import your Letterboxd ratings
  python recommend.py build                 Rebuild feature matrix from cache
  python recommend.py train                 Rebuild embeddings (re-run after new ratings)
  python recommend.py run     [--top N] [--temp T]  Show top N recommendations (default 20)
  python recommend.py add     <title> <year> <rating 1-10>  Add a single film manually
  python recommend.py list                              List all rated films
  python recommend.py del     <title> [year]            Delete a rating entry

Options:
  --top N   Number of recommendations to show (default: 20)
  --temp T  Temperature for result randomness (default: 0.0 = deterministic)
            0.0  identical results every run
            0.1  very mild shuffling, adjacent items may swap
            0.3  moderate exploration, good default for variety
            0.5  substantial reordering
            1.0+ extreme, not recommended

Examples:
  python recommend.py add "Dogville" "2003" 10
  python recommend.py run --top 40
  python recommend.py run --temp 0.3

First-time setup:
  1. Get a free TMDB API key: https://www.themoviedb.org/settings/api
  2. export TMDB_API_KEY=your_key
  3. python recommend.py setup              (downloads ~330 MB, then fetches metadata)
  4. python recommend.py ingest             (scrapes your Letterboxd profile)
  5. python recommend.py train              (builds embeddings)
  6. python recommend.py run                (prints recommendations)
"""
import sys
import json
from pathlib import Path

# make sure imports work from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import RATINGS_FILE


def cmd_setup(args):
    from src.catalog import download_movielens, fetch_all_metadata
    from src.features import build_features

    limit = None
    if '--limit' in args:
        limit = int(args[args.index('--limit') + 1])

    download_movielens()
    fetch_all_metadata(limit=limit)
    build_features()


def cmd_ingest(args):
    import time
    from src.letterboxd import scrape_ratings, read_csv_export
    from src.tmdb import TMDBClient

    if '--csv' in args:
        rated = read_csv_export()
    else:
        rated = scrape_ratings()

    if not rated:
        print("No ratings found.")
        return

    # Match each film to a TMDB ID
    print(f"\nMatching {len(rated)} films to TMDB ...")
    client = TMDBClient()

    # Load existing ratings so we merge rather than replace
    existing = {}
    if RATINGS_FILE.exists():
        with open(RATINGS_FILE) as f:
            existing = json.load(f)

    ratings = dict(existing)
    failed  = []
    new_count = 0

    for film in rated:
        tmdb_id = film.get('tmdb_id')

        if tmdb_id:
            tid_str = str(tmdb_id)
            client.get_movie(tmdb_id)
        else:
            result = client.search(film['title'], year=film.get('year'))
            if result:
                tid_str = str(result['id'])
                client.get_movie(result['id'])
            else:
                failed.append(film['title'])
                continue

        if tid_str not in ratings:
            new_count += 1
        ratings[tid_str] = film['rating']
        time.sleep(0.05)

    RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RATINGS_FILE, 'w') as f:
        json.dump(ratings, f, indent=2)

    print(f"\nTotal: {len(ratings)} films ({new_count} new)")
    if failed:
        print(f"Could not match {len(failed)} film(s):")
        for t in failed[:10]:
            print(f"  - {t}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    print(f"Saved → {RATINGS_FILE}")


def cmd_build(_args):
    from src.features import build_features
    build_features()


def cmd_run(args):
    from src.embeddings import EMBEDDINGS_FILE, recommend as emb_recommend, print_recommendations
    from src.nn_model import MODEL_FILE, load_model, predict as nn_predict
    from src.nn_features import FEATURES_FILE as NN_FEATURES_FILE, load_training_data

    top_n = 20
    if '--top' in args:
        top_n = int(args[args.index('--top') + 1])

    temperature = 0.0
    if '--temp' in args:
        temperature = float(args[args.index('--temp') + 1])

    if not EMBEDDINGS_FILE.exists():
        print("No embeddings. Run:  python recommend.py train")
        return

    with open(RATINGS_FILE) as f:
        ratings = json.load(f)

    # Optionally blend NN predictions if model + features are available
    nn_predictions = None
    if MODEL_FILE.exists() and NN_FEATURES_FILE.exists():
        _, _, X_all = load_training_data()
        model = load_model(X_all.shape[1])
        nn_predictions = nn_predict(model, X_all)

    recs = emb_recommend(ratings, top_n=top_n, nn_predictions=nn_predictions, temperature=temperature)
    print_recommendations(recs)


def cmd_train(args):
    """Train neural network on your ratings."""
    from src.embeddings import build_embeddings
    build_embeddings()
    print("Done. Run: python recommend.py run")


def cmd_add(args):
    """Add a single film by title, year, and rating."""
    if len(args) < 3:
        print("Usage: python recommend.py add <title> <year> <rating 1-10>")
        print("Example: python recommend.py add 'Casablanca' 1942 9")
        return

    title = args[0]
    year = args[1]
    try:
        score = int(args[2])
        if not (1 <= score <= 10):
            raise ValueError
    except ValueError:
        print(f"Rating must be 1-10, got: {args[2]}")
        return

    from src.tmdb import TMDBClient
    client = TMDBClient()

    result = client.search(title, year=year)
    if not result:
        print(f"Could not find '{title}' ({year}) on TMDB")
        return

    tid = str(result['id'])
    client.get_movie(result['id'])

    # Load existing
    try:
        with open(RATINGS_FILE) as f:
            ratings = json.load(f)
    except:
        ratings = {}

    ratings[tid] = score

    RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RATINGS_FILE, 'w') as f:
        json.dump(ratings, f, indent=2)

    title_found = result.get('title', title)
    print(f"✓ Added: {title_found} ({result.get('release_date', '')[:4]}) — {score}/10")
    print(f"Total ratings: {len(ratings)}")


def cmd_list(args):
    """List all rated films, sorted by rating descending."""
    if not RATINGS_FILE.exists():
        print("No ratings yet. Run: python recommend.py ingest")
        return

    with open(RATINGS_FILE) as f:
        ratings = json.load(f)

    from src.tmdb import TMDBClient
    client = TMDBClient()

    rows = []
    for tid_str, score in ratings.items():
        movie = client.get_movie(int(tid_str))
        if movie:
            title = movie.get('title', '?')
            rd = movie.get('release_date', '') or ''
            year = rd[:4] if len(rd) >= 4 else '????'
        else:
            title = f'[TMDB {tid_str}]'
            year = '????'
        rows.append((score, title, year))

    rows.sort(key=lambda r: (-r[0], r[2], r[1]))

    print(f"\n  {'#':>3}  {'Score':>5}  {'Year':>4}  Title")
    print(f"  {'─'*3}  {'─'*5}  {'─'*4}  {'─'*40}")
    for i, (score, title, year) in enumerate(rows, 1):
        print(f"  {i:>3}  {score:>4}/10  {year}  {title}")
    print(f"\n  Total: {len(rows)} films")


def cmd_del(args):
    """Delete a rated film by title (and optional year)."""
    if not args:
        print("Usage: python recommend.py del <title> [year]")
        return

    title_query = args[0].lower()
    year_query = args[1] if len(args) > 1 else None

    with open(RATINGS_FILE) as f:
        ratings = json.load(f)

    from src.tmdb import TMDBClient
    client = TMDBClient()

    matches = []
    for tid_str, score in ratings.items():
        movie = client.get_movie(int(tid_str))
        if not movie:
            continue
        title = movie.get('title', '')
        rd = (movie.get('release_date', '') or '')[:4]
        if title_query in title.lower():
            if year_query is None or rd == str(year_query):
                matches.append((tid_str, title, rd, score))

    if not matches:
        print(f"No match found for '{args[0]}'")
        return

    if len(matches) > 1:
        print("Multiple matches — be more specific (add year):")
        for tid_str, title, year, score in matches:
            print(f"  {title} ({year})  {score}/10  [id {tid_str}]")
        return

    tid_str, title, year, score = matches[0]
    del ratings[tid_str]

    with open(RATINGS_FILE, 'w') as f:
        json.dump(ratings, f, indent=2)

    print(f"Deleted: {title} ({year}) — {score}/10")
    print(f"Total ratings: {len(ratings)}")


COMMANDS = {
    'setup':  cmd_setup,
    'ingest': cmd_ingest,
    'build':  cmd_build,
    'train':  cmd_train,
    'run':    cmd_run,
    'add':    cmd_add,
    'list':   cmd_list,
    'del':    cmd_del,
}


def main():
    args = sys.argv[1:]
    if not args or args[0] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[args[0]](args[1:])


if __name__ == '__main__':
    main()
