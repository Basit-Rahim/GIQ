"""Loading the anonymised survey table and the distance-time lookup."""
import numpy as np
import pandas as pd

from src import config as C


def load_training_table() -> pd.DataFrame:
    return pd.read_csv(C.TRAINING_TABLE)


def minutes_for_km(km) -> np.ndarray:
    """Typical journey time for a distance: the survey's median reported time
    per distance band, interpolated."""
    lookup = pd.read_csv(C.DISTANCE_TIME_TABLE)
    return np.interp(np.asarray(km, dtype=float), lookup["km"], lookup["minutes"])
