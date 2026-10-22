# Household school-enrolment model: replication code

Replication code for a two-stage model of household school-enrolment decisions,
trained on a household survey in an urban community and tested on the Pakistan
Social and Living Standards Measurement (PSLM) 2019-20 survey for Lahore.

```
household inputs (income in PKR, distance in km, time in minutes)
  → Stage 1     five neural networks, one per parental perception (four-point scale)
  → autoencoder five perception scores → three latent dimensions
  → Stage 2     stacked ensemble (XGBoost, random forest, linear model; linear meta-learner)
                → share of the household's school-age children enrolled
  → route safety: probability × 0.8 if the route to school is unsafe (stated assumption)
```

## Folder structure

```
data/
  processed/households.csv            anonymised survey table (356 households)
  processed/distance_time_lookup.csv  median journey time per distance band
  raw/                                put the PSLM download here (not in git)
  interim/                            created by step 2 (not in git)
src/
  config.py      all paths and parameters
  data.py        loading the survey table and the distance-time lookup
  models.py      Stage 1, autoencoder, Stage 2, prediction
  evaluate.py    cross-validation
  pslm.py        PSLM download → Lahore household table
  validation.py  PSLM validation
scripts/         the four steps below
models/          trained model (recreated by step 1)
outputs/         all results (recreated by steps 1–4)
tests/
```

## Setup

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

**Survey table** (`data/processed/households.csv`). One row per surveyed urban
household with at least one child aged 5–16. It contains no names, identifiers,
coordinates, dates or place names; rows are in random order.

| Column | Meaning |
|---|---|
| `monthly_income` | household income, PKR per month (sum of members' reported income bands; 15 missing values imputed by k-nearest neighbours) |
| `travel_mode` | most common mode to school: 0 on foot, 1 bicycle, 2 motorcycle, 3 van/rickshaw, 4 public transport |
| `route_safe` | 1 if every child's route was reported safe, 0 otherwise |
| `read_write`, `solve_math` | household head can read and write / do simple sums (1 = yes) |
| `school_facilities` | weighted index of 11 facilities at the children's schools, 0–5 (sanitation and water weighted highest) |
| `min_distance`, `max_distance` | km to the nearest / farthest school among the household's children (band midpoints) |
| `min_time`, `max_time` | journey time in minutes (band midpoints) |
| `perceived_cost`, `perceived_distance`, `perceived_safety`, `perceived_facilities`, `stipend_helpfulness` | parents' answers, 0 (not important / not helpful) to 3 (very important / very helpful) |
| `enrolled_share` | share of the household's children aged 5–16 currently in school (model target) |
| `n_school_age` | number of children aged 5–16 |
| `school_info_imputed` | 1 if no child attends school, so the school-related answers were copied from the nearest surveyed household |

The survey asked distance, journey time, travel mode, route safety and school
facilities only about children who attend school. For the two households with no
child in school these answers were copied from the nearest surveyed household,
so filled values are real answers rather than an average that would reveal the
outcome. Because distance is recorded only for enrolled children, the model
cannot estimate a deterrent effect of distance.

**PSLM 2019-20** (not included). Download the PSLM 2019-20 microdata in Stata
format from the Pakistan Bureau of Statistics (pbs.gov.pk) and save the archive
as `data/raw/stata_data_pslm_2019_20.zip`. It must contain the folder
`stata data/` with `info.dta`, `roster.dta`, `secc1.dta` and `sece.dta`.

## Reproduce the results

Run from the repository root, in this order:

```bash
python scripts/train_model.py     # 1. cross-validation + final model (~1 min)
python scripts/build_pslm.py      # 2. PSLM download → Lahore household table
python scripts/validate_pslm.py   # 3. PSLM validation
python scripts/summarise.py       # 4. every number cited in the paper
python -m pytest tests -q         # checks
```

Seeds are fixed (`src/config.py`), so reruns give identical outputs.

## Where each result comes from

| Paper | Result | Script | Output |
|---|---|---|---|
| §3.2 | 356 households, 706 children, 98.7% enrolled | `summarise.py` | `outputs/summary.md` |
| §3.2.1 | about 3% of households report an unsafe route | `summarise.py` | `outputs/summary.md` |
| §3.2.2 | PSLM: 3,512 households, 7,929 children | `build_pslm.py` | `outputs/pslm_sample.json` |
| §3.2.2, §4.4 | 86.7% estimated vs 83.8% observed enrolment | `validate_pslm.py` | `outputs/pslm_validation.md` |
| §3.2.2 | AUC 0.67 | `validate_pslm.py` | `outputs/pslm_validation.md` |
| §6.3 | within three percentage points | `summarise.py` | `outputs/summary.md` |
| – | sensitivity: 76.7% (all routes unsafe) to 96.6% (all safe) | `validate_pslm.py` | `outputs/pslm_validation.md` |
| – | cross-validated model metrics | `train_model.py` | `outputs/model_metrics.md` |

## License

Code: MIT (see `LICENSE`). The survey table is provided for replication of the
paper's results only.
