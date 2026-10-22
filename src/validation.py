"""
External validation on PSLM 2019-20 Lahore households.

PSLM supplies each household's income and the head's literacy and numeracy.
Everything else is held at one stated value for every household, because PSLM
either lacks it or records it only for children who attend school:

    distance        0.1 km (nearest) to 1.0 km (farthest), a typical walkable
                    urban journey (see config.PSLM_MIN_DISTANCE_KM)
    journey time    survey median time for those distances
    travel mode     the most common mode in the training data
    facilities      the training-data mean
    route safety    equally likely safe or unsafe (config.PSLM_UNSAFE_ROUTE_SHARE)

Because these are constant, the test measures what the model learned from
income, literacy and numeracy - not the effect of distance or facilities.
"""
import numpy as np
import pandas as pd

from src import config as C
from src.evaluate import regression_metrics
from src.models import predict
from src.data import load_training_table, minutes_for_km


def held_values() -> dict:
    train = load_training_table()
    return {
        "travel_mode": int(train["travel_mode"].mode().min()),
        "school_facilities": round(float(train["school_facilities"].mean()), 2),
        "route_safe": 1,
        "min_distance": C.PSLM_MIN_DISTANCE_KM,
        "max_distance": C.PSLM_MAX_DISTANCE_KM,
        "min_time": round(float(minutes_for_km(C.PSLM_MIN_DISTANCE_KM)), 1),
        "max_time": round(float(minutes_for_km(C.PSLM_MAX_DISTANCE_KM)), 1),
    }


def pslm_inputs(households: pd.DataFrame) -> pd.DataFrame:
    inputs = households.copy()
    for col, value in held_values().items():
        inputs[col] = value
    return inputs


def _group_means(df: pd.DataFrame, by: pd.Series) -> dict:
    g = df.assign(group=by.values).groupby("group", observed=True)
    return {str(k): {"households": int(len(v)), "actual": round(float(v[C.TARGET_COLUMN].mean()), 3),
                     "model": round(float(v["enrollment_prob"].mean()), 3)} for k, v in g}


def validate(bundle: dict, households: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    preset = bundle["preset"]
    inputs = pslm_inputs(households)
    p_safe = predict(bundle, inputs.assign(route_safe=1))["enrollment_prob"].to_numpy()
    p_unsafe = predict(bundle, inputs.assign(route_safe=0))["enrollment_prob"].to_numpy()
    share = C.PSLM_UNSAFE_ROUTE_SHARE
    out = households.reset_index(drop=True).assign(enrollment_prob=(1 - share) * p_safe + share * p_unsafe)
    y, p = out[C.TARGET_COLUMN].to_numpy(), out["enrollment_prob"].to_numpy()

    metrics = regression_metrics(y, p)
    flagged = p < C.ENROLLED_THRESHOLD
    not_enrolled = y < C.ENROLLED_THRESHOLD
    metrics["flagged_at_0.5"] = int(flagged.sum())
    metrics["share_of_flagged_not_enrolled"] = float(not_enrolled[flagged].mean()) if flagged.any() else None
    metrics["share_not_enrolled_overall"] = float(not_enrolled.mean())
    metrics["accuracy_at_0.5"] = float(((p >= 0.5) == (y >= 0.5)).mean())
    metrics["accuracy_if_everyone_called_enrolled"] = float((y >= 0.5).mean())
    # Overall rate: enrolled children / all school-age children (predicted:
    # each household's probability times its number of children).
    n = out["n_school_age"].to_numpy()
    metrics["children_enrolled_rate_actual"] = float((y * n).sum() / n.sum())
    metrics["children_enrolled_rate_predicted"] = float((p * n).sum() / n.sum())
    # Sensitivity of the rate to the route-safety assumption
    metrics["children_enrolled_rate_predicted_all_routes_safe"] = float((p_safe * n).sum() / n.sum())
    metrics["children_enrolled_rate_predicted_all_routes_unsafe"] = float((p_unsafe * n).sum() / n.sum())

    quintile = pd.qcut(out["monthly_income"].rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    band = pd.qcut(pd.Series(p).rank(method="first"), 5, labels=[f"band {i}" for i in range(1, 6)])
    report = {
        "model": preset,
        "held_values": {**held_values(), "route_safe": f"{share:.0%} unsafe (expected value)"},
        "metrics": metrics,
        "by_income_quintile": _group_means(out, quintile),
        "by_head_literacy": _group_means(out, out["read_write"].map({0: "cannot read/write", 1: "can read/write"})),
        "by_head_numeracy": _group_means(out, out["solve_math"].map({0: "cannot do sums", 1: "can do sums"})),
        "by_predicted_band_low_to_high": _group_means(out, band),
    }
    return report, out
