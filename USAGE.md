# autofilm usage

## Prerequisites

```bash
export TMDB_API_KEY=your_key_here
```

Get a free key at https://www.themoviedb.org/settings/api

## Commands

### setup — download catalog and fetch metadata

```bash
python recommend.py setup
python recommend.py setup --limit 1000   # only fetch first 1000 films (for testing)
```

Downloads the MovieLens dataset (~330 MB), fetches TMDB metadata for all films, and builds the feature matrix. Run this once.

### ingest — import Letterboxd ratings

```bash
python recommend.py ingest               # scrape from RSS feed (recent diary entries)
python recommend.py ingest --csv         # import from Letterboxd CSV export
python recommend.py ingest --user alice  # use a named profile (Letterboxd @alice)
```

For CSV: go to Letterboxd > Settings > Import & Export > Export Your Data, then place `ratings.csv` in the project root.

Merges with existing ratings (never overwrites).

### add — rate a single film

```bash
python recommend.py add "Leviathan" "2014" 9
python recommend.py add "Stalker" "1979" 10
python recommend.py add "Casablanca" "" 9       # empty year if unsure
```

Searches TMDB, caches the metadata, and saves your rating (1-10).

### train — build semantic embeddings

```bash
python recommend.py train
```

Embeds all cached films using a sentence transformer (`all-MiniLM-L6-v2`). Uses GPU if available. Takes ~30s on GPU, a few minutes on CPU.

Run this after adding new films via `setup` or `add`.

### run — show recommendations

```bash
python recommend.py run                  # default: top 20, deterministic
python recommend.py run --top 50         # show top 50 instead of default 20
python recommend.py run --temp 0.3       # add randomness so repeated runs differ
python recommend.py run --top 30 --temp 0.3  # combine both flags
python recommend.py run --user alice     # run for a specific profile
```

### multi-user profiles

Use `--user <id>` to scope ratings and NN artifacts to a profile. Ratings and model files are stored under `data/users/<id>/`.

Examples:

```bash
python recommend.py ingest --user alice
python recommend.py train --user alice
python recommend.py run --user alice

python recommend.py ingest --user bob
python recommend.py run --user bob
```

**Temperature (`--temp`)** controls how much the ranking shuffles between runs:

| Value | Effect |
|-------|--------|
| 0.0 | Deterministic — identical results every run (default) |
| 0.1 | Very mild shuffling, adjacent items may swap |
| 0.3 | Moderate exploration, good default for variety |
| 0.5 | Substantial reordering |
| 1.0+ | Extreme, not recommended |

Each recommendation also includes a short explanation of why it was suggested, highlighting the strongest scoring factors (taste similarity, director affinity, obscurity, quality, prestige, locale).

### build — rebuild feature matrix (legacy)

```bash
python recommend.py build
```

Rebuilds the sparse feature matrix. Not needed for the current embedding-based approach.

## Typical workflow

First time:

```bash
python recommend.py setup
python recommend.py ingest --csv
python recommend.py train
python recommend.py run
```

Adding ratings and refreshing:

```bash
python recommend.py add "Blue Velvet" "1986" 10
python recommend.py add "Titanic" "1997" 4
python recommend.py train
python recommend.py run
```

Full pipeline in one shot:

```bash
python recommend.py ingest && python recommend.py train && python recommend.py run
```
