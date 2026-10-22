"""
The two-stage enrolment model.

    household inputs (raw units)
        -> Stage 1: five neural networks, one per perception (4-point scale);
                    each outputs the expected score sum_k k * P(k)
        -> autoencoder: five perception scores -> three latent dimensions z1-z3
        -> Stage 2: stacked ensemble (XGBoost + random forest + linear model,
                    linear meta-learner) on nine factual inputs + z1-z3
                    -> share of school-age children enrolled
        -> route-safety assumption: x 0.8 if the route is unsafe, clipped to [0, 1]

Everything needed to reproduce a prediction - scaling statistics, column order,
weights - is stored in one bundle, so training and inference can
never disagree on how inputs are prepared.
"""
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src import config as C

# The networks are tiny; one thread is ~5x faster than many and deterministic.
torch.set_num_threads(1)

BUNDLE_VERSION = 2
TARGETS = list(C.STAGE1_TARGETS)
LATENT_COLUMNS = [f"z{i + 1}" for i in range(C.LATENT_DIM)]


# Networks
class PerceptionNet(nn.Module):
    """Stage 1: household inputs -> logits over the four answer levels."""

    def __init__(self, n_inputs: int, hidden=C.STAGE1_HIDDEN, n_classes=C.N_LIKERT_CLASSES):
        super().__init__()
        layers, width = [], n_inputs
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(0.1)]
            width = h
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    """Five perception scores -> three latent dimensions -> five scores."""

    def __init__(self, n_inputs=len(TARGETS), hidden=C.AE_HIDDEN, latent=C.LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_inputs))

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Helpers
def _standardiser(x: np.ndarray):
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _tensor(x) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32))


def _class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights (add-one smoothed) so rare answers are not ignored."""
    counts = np.bincount(y, minlength=C.N_LIKERT_CLASSES) + 1.0
    return _tensor(counts.sum() / (C.N_LIKERT_CLASSES * counts))


def make_stage2(seed: int = C.SEED) -> StackingRegressor:
    return StackingRegressor(
        estimators=[
            ("xgboost", XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                     subsample=0.9, colsample_bytree=0.9, random_state=seed,
                                     n_jobs=1)),
            ("random_forest", RandomForestRegressor(n_estimators=300, max_depth=8,
                                                    min_samples_leaf=3, random_state=seed,
                                                    n_jobs=1)),
            ("linear", LinearRegression()),
        ],
        final_estimator=LinearRegression(),
        cv=KFold(C.CV_FOLDS, shuffle=True, random_state=seed),
    )


# Stage 1
def fit_stage1(X: np.ndarray, Y: np.ndarray, seed: int = C.SEED) -> dict:
    mean, std = _standardiser(X)
    xs = _tensor((X - mean) / std)
    states = {}
    for j, name in enumerate(TARGETS):
        torch.manual_seed(seed + j)
        y = Y[:, j].astype(int)
        model = PerceptionNet(X.shape[1])
        loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y))
        opt = torch.optim.Adam(model.parameters(), lr=C.STAGE1_LR, weight_decay=C.STAGE1_WEIGHT_DECAY)
        yt = torch.tensor(y)
        model.train()
        for _ in range(C.STAGE1_EPOCHS):
            opt.zero_grad()
            loss_fn(model(xs), yt).backward()
            opt.step()
        states[name] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {"mean": mean, "std": std, "states": states, "n_inputs": X.shape[1]}


def stage1_probabilities(stage1: dict, X: np.ndarray) -> dict:
    xs = _tensor((X - stage1["mean"]) / stage1["std"])
    out = {}
    for name in TARGETS:
        model = PerceptionNet(stage1["n_inputs"])
        model.load_state_dict(stage1["states"][name])
        model.eval()
        with torch.no_grad():
            out[name] = torch.softmax(model(xs), dim=1).numpy()
    return out


def stage1_scores(stage1: dict, X: np.ndarray) -> np.ndarray:
    """Expected answer level per perception, shape (n, 5), values in [0, 3]."""
    levels = np.arange(C.N_LIKERT_CLASSES)
    probs = stage1_probabilities(stage1, X)
    return np.column_stack([probs[name] @ levels for name in TARGETS])


# Autoencoder
def fit_autoencoder(scores: np.ndarray, seed: int = C.SEED) -> dict:
    """Trained on the observed answers, so the latent space describes how
    perceptions co-vary in the survey; applied to Stage 1 predictions."""
    torch.manual_seed(seed)
    mean, std = _standardiser(scores)
    xs = _tensor((scores - mean) / std)
    model = Autoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=C.AE_LR)
    loss_fn = nn.MSELoss()
    for _ in range(C.AE_EPOCHS):
        opt.zero_grad()
        loss = loss_fn(model(xs), xs)
        loss.backward()
        opt.step()
    return {"mean": mean, "std": std, "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "reconstruction_mse": float(loss.item())}


def encode(ae: dict, scores: np.ndarray) -> np.ndarray:
    model = Autoencoder()
    model.load_state_dict(ae["state"])
    model.eval()
    with torch.no_grad():
        return model.encoder(_tensor((scores - ae["mean"]) / ae["std"])).numpy()


# Full pipeline
def _out_of_fold_scores(X: np.ndarray, Y: np.ndarray, seed: int) -> np.ndarray:
    """Stage 1 predictions for the training rows from models that did not see
    them, so Stage 2 is trained on inputs like those it receives at inference."""
    oof = np.zeros((len(X), len(TARGETS)), dtype=np.float32)
    for k, (fit_idx, held_idx) in enumerate(KFold(C.CV_FOLDS, shuffle=True, random_state=seed).split(X)):
        oof[held_idx] = stage1_scores(fit_stage1(X[fit_idx], Y[fit_idx], seed + 100 * k), X[held_idx])
    return oof


def stage2_frame(df: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    frame = df[C.STAGE2_FACTUAL_COLUMNS].reset_index(drop=True).astype(float)
    return pd.concat([frame, pd.DataFrame(z, columns=LATENT_COLUMNS)], axis=1)


def fit_pipeline(df: pd.DataFrame, preset: str, seed: int = C.SEED,
                 return_stage2_inputs: bool = False):
    """Fit all stages. With return_stage2_inputs=True also returns the Stage 2
    training matrix (used to fit comparison models on identical inputs)."""
    X = df[C.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    Y = df[TARGETS].to_numpy(dtype=np.int64)
    y = df[C.TARGET_COLUMN].to_numpy(dtype=float)

    stage1 = fit_stage1(X, Y, seed)
    ae = fit_autoencoder(Y.astype(np.float32), seed)
    z_train = encode(ae, _out_of_fold_scores(X, Y, seed))
    stage2_X = stage2_frame(df, z_train)
    stage2 = make_stage2(seed).fit(stage2_X, y)

    bundle = {
        "version": BUNDLE_VERSION,
        "preset": preset,
        "feature_columns": list(C.FEATURE_COLUMNS),
        "stage1_targets": dict(C.STAGE1_TARGETS),
        "stage1": stage1,
        "autoencoder": ae,
        "stage2": stage2,
        "stage2_columns": list(stage2_X.columns),
        "unsafe_route_multiplier": C.UNSAFE_ROUTE_MULTIPLIER,
        "training_rows": len(df),
        "training_data_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest(),
        "seed": seed,
    }
    return (bundle, stage2_X) if return_stage2_inputs else bundle


def predict(bundle: dict, inputs: pd.DataFrame) -> pd.DataFrame:
    """Inputs in raw units (income in PKR, km, minutes). Returns perception
    scores, latent dimensions, the model output and the final probability."""
    inputs = inputs.reset_index(drop=True)
    X = inputs[bundle["feature_columns"]].to_numpy(dtype=np.float32)
    scores = stage1_scores(bundle["stage1"], X)
    z = encode(bundle["autoencoder"], scores)
    model_prob = np.clip(bundle["stage2"].predict(stage2_frame(inputs, z)[bundle["stage2_columns"]]), 0, 1)
    unsafe = inputs["route_safe"].fillna(1).astype(int).to_numpy() == 0
    final = np.clip(np.where(unsafe, model_prob * bundle["unsafe_route_multiplier"], model_prob), 0, 1)

    out = pd.DataFrame(scores, columns=TARGETS)
    out[LATENT_COLUMNS] = z
    out["model_prob"] = model_prob
    out["enrollment_prob"] = final
    return out
