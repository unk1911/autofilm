#!/usr/bin/env python3
"""
autofilm — personal movie recommendation engine

Usage:
  python recommend.py setup   [--limit N]   Download catalog + fetch TMDB metadata
  python recommend.py ingest  [--csv]       Import your Letterboxd ratings
  python recommend.py build                 Rebuild feature matrix from cache
  python recommend.py train                 Rebuild embeddings (re-run after new ratings)
  python recommend.py run     [--top N] [--temp T]  Show top N recommendations (default 20)
  python recommend.py similar <title> <year> [--top N]  Find films similar to a given film
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
  --user U  Profile id to use (default: default)

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

from src.config import LETTERBOXD_USERNAME
from src.user_paths import get_user_paths


def cmd_setup(args, _paths):
    from src.catalog import download_movielens, fetch_all_metadata
    from src.features import build_features

    limit = None
    if '--limit' in args:
        limit = int(args[args.index('--limit') + 1])

    download_movielens()
    fetch_all_metadata(limit=limit)
    build_features()


def cmd_ingest(args, paths):
    import time
    from src.letterboxd import scrape_ratings, read_csv_export
    from src.tmdb import TMDBClient

    if '--csv' in args:
        rated = read_csv_export()
    else:
        username = LETTERBOXD_USERNAME if paths.user == 'default' else paths.user
        rated = scrape_ratings(username=username)

    if not rated:
        print("No ratings found.")
        return

    # Match each film to a TMDB ID
    print(f"\nMatching {len(rated)} films to TMDB ...")
    client = TMDBClient()

    # Load existing ratings so we merge rather than replace
    existing = {}
    if paths.ratings_file.exists():
        with open(paths.ratings_file) as f:
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

    paths.ratings_file.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.ratings_file, 'w') as f:
        json.dump(ratings, f, indent=2)

    print(f"\nTotal: {len(ratings)} films ({new_count} new)")
    if failed:
        print(f"Could not match {len(failed)} film(s):")
        for t in failed[:10]:
            print(f"  - {t}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    print(f"Saved ({paths.user}) → {paths.ratings_file}")


def cmd_build(_args, _paths):
    from src.features import build_features
    build_features()


def cmd_run(args, paths):
    from src.embeddings import EMBEDDINGS_FILE, recommend as emb_recommend, print_recommendations
    from src.nn_model import load_model, predict as nn_predict
    from src.nn_features import load_training_data

    top_n = 20
    if '--top' in args:
        top_n = int(args[args.index('--top') + 1])

    temperature = 0.0
    if '--temp' in args:
        temperature = float(args[args.index('--temp') + 1])

    if not EMBEDDINGS_FILE.exists():
        print("No embeddings. Run:  python recommend.py train")
        return

    if not paths.ratings_file.exists():
        print(f"No ratings found for '{paths.user}'. Run: python recommend.py ingest --user {paths.user}")
        return

    with open(paths.ratings_file) as f:
        ratings = json.load(f)

    # Optionally blend NN predictions if model + features are available
    nn_predictions = None
    if paths.nn_model_file.exists() and paths.nn_features_file.exists():
        _, _, X_all = load_training_data(features_file=paths.nn_features_file)
        model = load_model(X_all.shape[1], model_file=paths.nn_model_file)
        nn_predictions = nn_predict(model, X_all)

    recs = emb_recommend(ratings, top_n=top_n, nn_predictions=nn_predictions, temperature=temperature)
    print_recommendations(recs)


def cmd_similar(args, paths):
    """Find films similar to a given title, personalized by your taste."""
    if len(args) < 2:
        print("Usage: python recommend.py similar <title> <year> [--top N]")
        print("Example: python recommend.py similar 'I Am Still Here' 2024")
        return

    title = args[0]
    year = args[1]

    top_n = 20
    if '--top' in args:
        top_n = int(args[args.index('--top') + 1])

    from src.embeddings import find_similar, print_recommendations

    if not paths.ratings_file.exists():
        print("No ratings found. Run: python recommend.py ingest")
        return

    with open(paths.ratings_file) as f:
        ratings = json.load(f)

    recs = find_similar(title, year, ratings, top_n=top_n)
    if not recs:
        print("No similar films found.")
        return

    print(f"\n  Similar to: {title} ({year})")
    print_recommendations(recs)


def cmd_train(args, paths):
    """Train embeddings and user-specific NN artifacts."""
    from src.embeddings import build_embeddings
    from src.nn_features import build_training_data
    from src.nn_model import train_model

    build_embeddings()

    if not paths.ratings_file.exists():
        print(f"No ratings found for '{paths.user}'. Skipping NN training.")
        print(f"Add ratings first: python recommend.py ingest --user {paths.user}")
        print(f"Done. Run: python recommend.py run --user {paths.user}")
        return

    with open(paths.ratings_file) as f:
        ratings = json.load(f)

    if not ratings:
        print(f"No ratings found for '{paths.user}'. Skipping NN training.")
        print(f"Done. Run: python recommend.py run --user {paths.user}")
        return

    paths.user_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train, *_ = build_training_data(
        ratings,
        features_file=paths.nn_features_file,
        vocab_file=paths.nn_vocab_file,
    )
    train_model(X_train, y_train, model_file=paths.nn_model_file)
    print(f"Done. Run: python recommend.py run --user {paths.user}")


def cmd_add(args, paths):
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
        with open(paths.ratings_file) as f:
            ratings = json.load(f)
    except:
        ratings = {}

    ratings[tid] = score

    paths.ratings_file.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.ratings_file, 'w') as f:
        json.dump(ratings, f, indent=2)

    title_found = result.get('title', title)
    print(f"✓ Added: {title_found} ({result.get('release_date', '')[:4]}) — {score}/10")
    print(f"Total ratings: {len(ratings)}")


def cmd_list(args, paths):
    """List all rated films, sorted by rating descending."""
    if not paths.ratings_file.exists():
        print("No ratings yet. Run: python recommend.py ingest")
        return

    with open(paths.ratings_file) as f:
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


def cmd_del(args, paths):
    """Delete a rated film by title (and optional year)."""
    if not args:
        print("Usage: python recommend.py del <title> [year]")
        return

    title_query = args[0].lower()
    year_query = args[1] if len(args) > 1 else None

    if not paths.ratings_file.exists():
        print("No ratings yet. Run: python recommend.py ingest")
        return

    with open(paths.ratings_file) as f:
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

    with open(paths.ratings_file, 'w') as f:
        json.dump(ratings, f, indent=2)

    print(f"Deleted: {title} ({year}) — {score}/10")
    print(f"Total ratings: {len(ratings)}")


COMMANDS = {
    'setup':   cmd_setup,
    'ingest':  cmd_ingest,
    'build':   cmd_build,
    'train':   cmd_train,
    'run':     cmd_run,
    'similar': cmd_similar,
    'add':     cmd_add,
    'list':    cmd_list,
    'del':     cmd_del,
}


def main():
    args = sys.argv[1:]

    user = 'default'
    if '--user' in args:
        i = args.index('--user')
        if i + 1 >= len(args):
            raise SystemExit("Missing value for --user")
        user = args[i + 1]
        args = args[:i] + args[i + 2:]

    try:
        paths = get_user_paths(user)
    except ValueError as e:
        raise SystemExit(str(e))

    if not args or args[0] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    print(f"Using profile: {paths.user}")
    COMMANDS[args[0]](args[1:], paths)


if __name__ == '__main__':
    main()
