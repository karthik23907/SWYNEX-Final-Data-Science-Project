"""Stateless feature engineering shared by the modeling notebook and the saved model.

Kept in a module (not in the notebook) so that `models/churn_model.joblib` can be
loaded from any script: pickle stores a reference to `src.features.add_features`.
"""
from __future__ import annotations

import pandas as pd

ADDONS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Add `n_addons`, the number of add-on services a customer has (0-6)."""
    d = d.copy()
    d["n_addons"] = d[ADDONS].sum(axis=1)
    return d
