"""
Step 3 - Validate the trained model on PSLM 2019-20 Lahore households.

    python scripts/validate_pslm.py

Reads  models/model.joblib, data/interim/pslm_lahore_households.csv
Writes outputs/pslm_validation.json, outputs/pslm_validation.md
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.validation import validate


def _pct(v):
    return "–" if v is None else f"{v:.1%}"


def _num(v):
    return "–" if v is None else f"{v:.3f}"


def write_markdown(r: dict, sample: dict) -> str:
    m = r["metrics"]
    lines = ["# PSLM 2019-20 Lahore validation", "",
             f"{sample['households_used']} households with school-age children; "
             f"{sample['school_age_children']} children aged 5–16.", "",
             "Income, head literacy and head numeracy come from PSLM. Distance, journey time, travel mode and "
             "facilities are held at one value for every household; route safety is equally likely safe or "
             "unsafe (see Held values).", "",
             "## Overall enrolment rate (enrolled children / all children aged 5–16)", "",
             "| | Rate |", "|---|---|",
             f"| PSLM observed | {_pct(m['children_enrolled_rate_actual'])} |",
             f"| Model, routes 50% unsafe (reported) | {_pct(m['children_enrolled_rate_predicted'])} |",
             f"| Model, all routes safe | {_pct(m['children_enrolled_rate_predicted_all_routes_safe'])} |",
             f"| Model, all routes unsafe | {_pct(m['children_enrolled_rate_predicted_all_routes_unsafe'])} |", "",
             "## Household level", "",
             "| AUC | Mean pred. / actual | Flagged @0.5 | Not-enrolled found | Balanced acc. @0.5 | "
             "Accuracy @0.5 (all-enrolled guess) |",
             "|---|---|---|---|---|---|",
             f"| {_num(m['auc'])} | {_num(m['mean_predicted'])} / {_num(m['mean_actual'])} | {m['flagged_at_0.5']} | "
             f"{m['not_enrolled_found_at_0.5']} / {m['not_enrolled_total']} | {_num(m['balanced_accuracy_at_0.5'])} | "
             f"{_pct(m['accuracy_at_0.5'])} ({_pct(m['accuracy_if_everyone_called_enrolled'])}) |"]
    for section, title in [("by_income_quintile", "By income quintile (PSLM's own)"),
                           ("by_head_literacy", "By head literacy"),
                           ("by_head_numeracy", "By head numeracy"),
                           ("by_predicted_band_low_to_high", "Calibration: predicted fifths, low to high")]:
        groups = list(r[section])
        lines += ["", f"## {title}", "", "| | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups),
                  "| Model | " + " | ".join(f"{r[section][g]['model']:.2f}" for g in groups) + " |",
                  "| Actual | " + " | ".join(f"{r[section][g]['actual']:.2f}" for g in groups) + " |"]
    lines += ["", "## Held values", ""] + [f"- {k} = {v}" for k, v in r["held_values"].items()]
    return "\n".join(lines) + "\n"


def main():
    if not C.PSLM_TABLE.exists():
        sys.exit("Run scripts/build_pslm.py first.")
    report, _ = validate(joblib.load(C.MODEL_BUNDLE), pd.read_csv(C.PSLM_TABLE))
    sample = json.loads((C.OUTPUTS_DIR / "pslm_sample.json").read_text())
    (C.OUTPUTS_DIR / "pslm_validation.json").write_text(json.dumps(report, indent=2))
    (C.OUTPUTS_DIR / "pslm_validation.md").write_text(write_markdown(report, sample), encoding="utf-8")
    print(f"Children enrolled: predicted {report['metrics']['children_enrolled_rate_predicted']:.1%}, "
          f"observed {report['metrics']['children_enrolled_rate_actual']:.1%}; AUC {report['metrics']['auc']:.3f}")


if __name__ == "__main__":
    main()
