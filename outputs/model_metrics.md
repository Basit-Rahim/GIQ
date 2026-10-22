# Model metrics (cross-validated)

Repeated stratified k-fold (2 folds, 3 repeats); the whole pipeline is refitted in every fold.
Binary metrics treat a household as enrolled when at least half of its school-age children attend.

356 households, 2 not enrolled.

## Stage 1: perceptions (held-out households)

| Perception | Item | Accuracy | Always-most-common | Balanced acc. | Macro-F1 | Spearman |
|---|---|---|---|---|---|---|
| perceived_cost | b2_1 | 60.4% | 87.1% | 34.7% | 0.315 | 0.066 |
| perceived_distance | b2_3 | 38.5% | 32.3% | 37.7% | 0.358 | 0.442 |
| perceived_safety | b2_4 | 66.8% | 85.1% | 38.5% | 0.316 | 0.286 |
| perceived_facilities | b2_14 | 53.8% | 82.0% | 34.7% | 0.276 | 0.158 |
| stipend_helpfulness | b3_1 | 54.5% | 82.3% | 34.2% | 0.319 | 0.136 |

## Stage 2: enrolled share (held-out households)

| Model | MAE | R² | AUC | Balanced acc. @0.5 | Not-enrolled found @0.5 | Mean pred. / actual |
|---|---|---|---|---|---|---|
| full_model | 0.029 | -0.126 | 0.529 | 0.500 | 0.0 / 2 | 0.983 / 0.985 |
| without_perceptions | 0.031 | -0.317 | 0.581 | 0.498 | 0.0 / 2 | 0.982 / 0.985 |
| xgboost_only | 0.027 | -0.502 | 0.622 | 0.497 | 0.0 / 2 | 0.985 / 0.985 |
| random_forest_only | 0.027 | -0.140 | 0.525 | 0.500 | 0.0 / 2 | 0.987 / 0.985 |
| linear_only | 0.031 | -0.114 | 0.355 | 0.500 | 0.0 / 2 | 0.982 / 0.985 |
| mean_baseline | 0.028 | -0.011 | 0.500 | 0.500 | 0.0 / 2 | 0.985 / 0.985 |
