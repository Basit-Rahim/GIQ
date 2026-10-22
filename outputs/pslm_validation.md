# PSLM 2019-20 Lahore validation

3512 households with school-age children; 7929 children aged 5–16.

Income, head literacy and head numeracy come from PSLM. Distance, journey time, travel mode and facilities are held at one value for every household; route safety is equally likely safe or unsafe (see Held values).

## Overall enrolment rate (enrolled children / all children aged 5–16)

| | Rate |
|---|---|
| PSLM observed | 83.8% |
| Model, routes 50% unsafe (reported) | 86.7% |
| Model, all routes safe | 96.6% |
| Model, all routes unsafe | 76.7% |

## Household level

| AUC | Mean pred. / actual | Flagged @0.5 | Not-enrolled found | Balanced acc. @0.5 | Accuracy @0.5 (all-enrolled guess) |
|---|---|---|---|---|---|
| 0.670 | 0.868 / 0.844 | 0 | 0 / 401 | 0.500 | 88.6% (88.6%) |

## By income quintile (PSLM's own)

| | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| Model | 0.86 | 0.86 | 0.87 | 0.87 | 0.88 |
| Actual | 0.81 | 0.81 | 0.85 | 0.86 | 0.90 |

## By head literacy

| | can read/write | cannot read/write |
|---|---|---|
| Model | 0.88 | 0.84 |
| Actual | 0.89 | 0.73 |

## By head numeracy

| | can do sums | cannot do sums |
|---|---|---|
| Model | 0.88 | 0.80 |
| Actual | 0.86 | 0.74 |

## Calibration: predicted fifths, low to high

| | band 1 | band 2 | band 3 | band 4 | band 5 |
|---|---|---|---|---|---|
| Model | 0.83 | 0.87 | 0.88 | 0.88 | 0.89 |
| Actual | 0.73 | 0.81 | 0.86 | 0.92 | 0.90 |

## Held values

- travel_mode = 0
- school_facilities = 3.58
- route_safe = 50% unsafe (expected value)
- min_distance = 0.1
- max_distance = 1.0
- min_time = 7.5
- max_time = 10.0
