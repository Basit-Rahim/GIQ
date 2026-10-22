"""
Step 1 - Cross-validate the model, then train it on all households.

    python scripts/train_model.py

Reads  data/processed/households.csv
Writes models/model.joblib
       outputs/metrics.json, outputs/model_metrics.md
"""
import argparse
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.data import load_training_table
from src.evaluate import cross_validate
from src.models import fit_pipeline


def _fmt(v, pct=False):
    if v is None:
        return "–"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def write_markdown(r: dict) -> str:
    lines = ["# Model metrics (cross-validated)", "",
             f"Repeated stratified k-fold ({r['folds']} folds, {r['repeats']} repeats); "
             "the whole pipeline is refitted in every fold.",
             "Binary metrics treat a household as enrolled when at least half of its school-age children attend.", "",
             f"{r['households']} households, {r['not_enrolled_households']} not enrolled.", "",
             "## Stage 1: perceptions (held-out households)", "",
             "| Perception | Item | Accuracy | Always-most-common | Balanced acc. | Macro-F1 | Spearman |",
             "|---|---|---|---|---|---|---|"]
    for name, m in r["stage1"].items():
        lines.append(f"| {name} | {m['survey_item']} | {_fmt(m['accuracy'], True)} | "
                     f"{_fmt(m['majority_class_accuracy'], True)} | {_fmt(m['balanced_accuracy'], True)} | "
                     f"{_fmt(m['macro_f1'])} | {_fmt(m['spearman_expected_score'])} |")
    lines += ["", "## Stage 2: enrolled share (held-out households)", "",
              "| Model | MAE | R² | AUC | Balanced acc. @0.5 | Not-enrolled found @0.5 | Mean pred. / actual |",
              "|---|---|---|---|---|---|---|"]
    for name, m in r["stage2"].items():
        found = f"{m['not_enrolled_found_at_0.5']:.1f} / {m['not_enrolled_total']:.0f}"
        lines.append(f"| {name} | {_fmt(m['mae'])} | {_fmt(m['r2'])} | {_fmt(m['auc'])} | "
                     f"{_fmt(m['balanced_accuracy_at_0.5'])} | {found} | "
                     f"{_fmt(m['mean_predicted'])} / {_fmt(m['mean_actual'])} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true", help="train the final model only")
    args = parser.parse_args()

    C.MODELS_DIR.mkdir(exist_ok=True)
    C.OUTPUTS_DIR.mkdir(exist_ok=True)
    df = load_training_table()

    if not args.skip_cv:
        print(f"Cross-validating on {len(df)} households ...", flush=True)
        results = cross_validate(df, C.PRESET)
        (C.OUTPUTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
        (C.OUTPUTS_DIR / "model_metrics.md").write_text(write_markdown(results), encoding="utf-8")

    print(f"Training the final model on all {len(df)} households ...", flush=True)
    joblib.dump(fit_pipeline(df, C.PRESET), C.MODEL_BUNDLE, compress=3)
    print(f"Saved {C.MODEL_BUNDLE.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
