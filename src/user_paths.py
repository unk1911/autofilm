import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR, RATINGS_FILE


DEFAULT_USER = 'default'


@dataclass(frozen=True)
class UserPaths:
    user: str
    user_dir: Path
    ratings_file: Path
    nn_features_file: Path
    nn_vocab_file: Path
    nn_model_file: Path


def normalize_user(user: str | None) -> str:
    raw = (user or '').strip()
    if not raw:
        return DEFAULT_USER
    if not re.fullmatch(r'[A-Za-z0-9._-]+', raw):
        raise ValueError(
            "Invalid user id. Use letters, numbers, '.', '_', or '-'."
        )
    return raw


def get_user_paths(user: str | None = None) -> UserPaths:
    normalized = normalize_user(user)
    user_dir = DATA_DIR / 'users' / normalized

    paths = UserPaths(
        user=normalized,
        user_dir=user_dir,
        ratings_file=user_dir / 'ratings.json',
        nn_features_file=user_dir / 'nn_features.npz',
        nn_vocab_file=user_dir / 'nn_vocab.npz',
        nn_model_file=user_dir / 'recommendation_model.pt',
    )

    _migrate_legacy_ratings_if_needed(paths)
    return paths


def _migrate_legacy_ratings_if_needed(paths: UserPaths) -> None:
    if paths.user != DEFAULT_USER:
        return
    if paths.ratings_file.exists() or not RATINGS_FILE.exists():
        return

    paths.ratings_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RATINGS_FILE, paths.ratings_file)
    print(f"Migrated legacy ratings → {paths.ratings_file}")
