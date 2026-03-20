"""Trainer registry — single source of truth for all available model trainers.

Usage::

    from backend.ml.training import get_trainer, list_trainers

    trainer = get_trainer("extra_trees")   # by key
    all_keys = list_trainers()             # ["extra_trees", "lightgbm"]

Adding a new model:
    1. Create ``backend/ml/training/my_model.py`` with a ``Trainer`` subclass
    2. Register it in ``_REGISTRY`` below
    3. Done — CLI, tuning, registry, and DagsHub all work automatically.
"""

from __future__ import annotations

from backend.ml.training.base import Trainer
from backend.ml.training.extra_tree import ExtraTreesTrainer
from backend.ml.training.lightgbm_trainer import LightGBMTrainer

_REGISTRY: dict[str, type[Trainer]] = {
    "extra_trees": ExtraTreesTrainer,
    "lightgbm": LightGBMTrainer,
}


def get_trainer(key: str) -> Trainer:
    """Instantiate a trainer by its registry key.

    Raises ``KeyError`` with a helpful message when *key* is unknown.
    """
    try:
        cls = _REGISTRY[key]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown trainer '{key}'. Available: {available}") from None
    return cls()


def list_trainers() -> list[str]:
    """Return sorted list of registered trainer keys."""
    return sorted(_REGISTRY)
