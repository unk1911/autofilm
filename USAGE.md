# sim-film usage

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
python recommend.py run
python recommend.py run --top 50         # show top 50 instead of default 20
```

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
