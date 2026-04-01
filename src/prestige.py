import json
from pathlib import Path
from .config import DATA_DIR

PRESTIGE_FILE = DATA_DIR / 'prestige_index.json'

_cache: dict[str, float] | None = None


def load_prestige_index() -> dict[str, float]:
    """Load prestige index from disk. Returns empty dict if file doesn't exist."""
    global _cache
    if _cache is not None:
        return _cache
    if not PRESTIGE_FILE.exists():
        _cache = {}
        return _cache
    with open(PRESTIGE_FILE) as f:
        _cache = json.load(f)
    return _cache


def prestige_score(tmdb_id: int) -> float:
    """Return prestige multiplier for a TMDB film ID (1.0 if not in index)."""
    return load_prestige_index().get(str(tmdb_id), 1.0)
