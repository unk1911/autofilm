# autofilm

A personal movie recommendation engine that learns your taste from rated films and surfaces obscure, art house, and world cinema you're likely to love.

## How it works

The system is content-based: it doesn't rely on what other users liked (collaborative filtering), but instead builds a semantic understanding of *your* taste by analyzing the films you've rated.

### Data pipeline

```
MovieLens catalog (~62k films)
        |
        v
  TMDB API metadata ──> SQLite cache (~85k films, ~1.3 GB)
        |                 (genres, directors, keywords, plot overviews,
        |                  countries, languages, vote stats)
        v
  Sentence Transformer (all-MiniLM-L6-v2)
        |
        v
  384-dim embedding per film ──> embeddings.npz
```

1. **Catalog**: MovieLens `ml-latest` provides a broad base of ~62k films with TMDB ID mappings.
2. **Metadata**: Each film is enriched via the TMDB API (genres, director, cast, keywords, plot overview, production countries, language, vote stats). Results are cached in SQLite so the API is only hit once per film.
3. **Embeddings**: A sentence transformer encodes each film into a 384-dimensional vector. The input text is a concatenation of: title, genres, director, plot overview, keywords, country/language, and year. This captures semantic meaning -- films with similar themes, styles, and contexts end up close together in vector space.

### Training (`recommend.py train`)

"Training" in this system means encoding all ~85k cached films into dense vector embeddings using `all-MiniLM-L6-v2` (a pretrained sentence transformer from HuggingFace). This is not traditional supervised learning -- the model is pretrained on a large text corpus and we use it as a fixed feature extractor.

For each film, we build a text description:

```
Mulholland Drive Genres: Thriller, Drama, Mystery. Directed by David Lynch.
After a car wreck on the winding Mulholland Drive renders a woman amnesiac...
Keywords: hollywood, los angeles, dream, identity, surreal...
Country: US, FR. Language: en. Year: 2001.
```

The sentence transformer maps this to a 384-dim unit vector. Films with semantically similar descriptions (themes, directors, moods, settings) have high cosine similarity between their vectors.

The embedding step takes ~30 seconds on GPU, a few minutes on CPU.

### Recommendation (`recommend.py run`)

At recommendation time, the system builds a **taste profile** from your ratings:

```
                    your 9-10 rated films
                           |
                    weighted mean of their embeddings (10s weighted 2x)
                           |
                      positive vector
                           |
taste profile = positive - 0.8 * negative
                           |
                      negative vector
                           |
                    weighted mean of their embeddings
                           |
                    your 1-5 rated films
```

Ratings are split into three bands:
- **9-10** (loved): form the positive taste vector. 10s are weighted 2x vs 9s.
- **6-8** (neutral): ignored entirely. These dilute the signal.
- **1-5** (disliked): form the negative taste vector, weighted by how much you disliked them (a 3/10 pushes harder than a 5/10).

The final taste profile is `positive - 0.8 * negative`, normalized to unit length.

### Scoring

Every unrated film is scored by:

```
score = similarity^2 * quality * obscurity_boost * locale_boost * director_boost * prestige
```

- **similarity**: cosine similarity between the film's embedding and your taste profile. Squared so that taste match dominates.
- **quality**: `(tmdb_score / 10)^1.5` -- rewards critically acclaimed films, penalizes mediocre ones.
- **obscurity_boost**: inverse log of vote count. Films with fewer votes get boosted; blockbusters with 5k+ votes are penalized. This surfaces hidden gems over obvious picks.
- **locale_boost**: small multiplier for preferred languages/countries (French, Korean, Russian, Croatian, Spanish, etc.).
- **director_boost**: up to 2.2x boost for directors with multiple loved films (score >= 9).
- **prestige**: multiplier (1.0-1.3x) for films recognized across major awards and curated lists (Sight & Sound, TSPDT, Criterion, Ebert).

### Filters

Before scoring, films are filtered out if they:
- Have fewer than 300 TMDB votes (removes fan-inflated scores)
- Score below 7.0 on TMDB (quality floor)
- Are shorter than 60 minutes (removes shorts, unless Japanese animation)
- Are primarily Family, TV Movie, or Romance genre
- Have cosine similarity below 0.25 (no weak matches)

After scoring, a **genre diversity cap** ensures no single genre takes more than 40% of the results.

### Temperature (result randomness)

By default, recommendations are deterministic -- the same ratings always produce the same list. The `--temp` flag introduces controlled randomness so you can explore different suggestions across runs:

```bash
python recommend.py run --temp 0.3
```

The system applies multiplicative log-normal noise to scores before ranking: `score *= exp(temperature * N(0,1))`. This preserves the ratio structure of scores -- high-quality matches still dominate, but nearby items shuffle position.

| Temperature | Effect |
|-------------|--------|
| 0.0 | Deterministic (default) |
| 0.1 | Very mild shuffling, adjacent items may swap |
| 0.3 | Moderate exploration, good default for variety |
| 0.5 | Substantial reordering |
| 1.0+ | Extreme, not recommended |

### Match explanations

Each recommendation includes a short paragraph explaining why it was suggested. The explanation highlights the strongest scoring factors for that film:

- Taste profile alignment (similarity strength)
- Director affinity (you've loved other films by this director)
- Obscurity (hidden gem with few ratings)
- Critical quality (high TMDB score)
- Prestige (recognized across film awards and curated lists)
- Locale preference (matches your preferred language/country)

## Architecture

```
recommend.py              CLI entrypoint (setup, ingest, add, train, run)
src/
  config.py               Constants, paths, language/country boosts
  catalog.py              MovieLens download + TMDB metadata fetch
  tmdb.py                 TMDB API client with SQLite cache
  letterboxd.py           Letterboxd RSS scraper + CSV import
  embeddings.py           Sentence transformer embeddings + recommendation engine
  prestige.py             Prestige score lookup from curated film lists
  features.py             (legacy) Sparse IDF-weighted feature vectors
  nn_model.py             (legacy) PyTorch neural network
  nn_features.py          (legacy) One-hot feature engineering for NN
data/
  tmdb_cache.db           SQLite cache of TMDB metadata
  users/<id>/ratings.json Per-user ratings (TMDB ID -> score 1-10)
  users/<id>/nn_*.npz     Per-user NN features/vocab artifacts
  users/<id>/recommendation_model.pt  Per-user NN model
  embeddings.npz          Precomputed 384-dim embeddings for all films
  prestige_index.json     Film prestige multipliers from awards/curated lists
```

Use `--user <id>` on CLI commands to switch profiles (default: `default`).

## Why not collaborative filtering?

Collaborative filtering (e.g., "users who liked X also liked Y") requires millions of user-item interactions to work well. For a personal recommendation engine with one user, content-based is the only viable approach. The sentence transformer embeddings capture richer semantic relationships than traditional one-hot features -- "surrealist French thriller" and "Lynchian mystery" end up nearby in vector space even if they share no explicit genre or keyword tags.
