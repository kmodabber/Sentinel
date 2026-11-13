import pandas as pd
from features import FEATURES


class MissingFeaturesError(ValueError):
  """The flows lack columns the model needs."""


def select_features(flows: pd.DataFrame) -> pd.DataFrame:
  """Return the model's feature columns, in the model's order."""
  missing = [c for c in FEATURES if c not in flows.columns]
  if missing:
      raise MissingFeaturesError(f"flows are missing {len(missing)} columns: {missing}")
  return flows[FEATURES].copy()