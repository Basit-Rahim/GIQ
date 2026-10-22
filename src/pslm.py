"""
Raw PSLM 2019-20 (PBS release) -> Lahore households for external validation.

Definitions:
    sample      usual household members (sb1q11 = 1) aged 5-16 with an education record
    outcome     share of those children currently attending school (sc1q01 = 3)
    income      household sum of earnings, transfers, remittances, rent and other
                income (sece), annual -> monthly; households whose income is
                unmeasured (zero but someone worked) or unanswered are dropped
    head        literacy = can read and write (sc1q1a, sc1q2a); numeracy = sc1q3a

PSLM records distance to school, school transport and its cost only for
children who attend, so those variables are NOT taken from PSLM (they would
reveal the outcome). The validation script holds them at stated values.
"""
import zipfile

import numpy as np
import pandas as pd
import pyreadstat

from src import config as C

MODULES = ("info", "roster", "secc1", "sece")


def _extract(zip_path, out_dir) -> str:
    folder = out_dir / "pslm_2019_20"
    with zipfile.ZipFile(zip_path) as z:
        for module in MODULES:
            if not (folder / "stata data" / f"{module}.dta").exists():
                z.extract(f"stata data/{module}.dta", folder)
    return str(folder / "stata data")


def _read(folder, module, cols=None) -> pd.DataFrame:
    df, _ = pyreadstat.read_dta(f"{folder}/{module}.dta", usecols=cols, encoding="latin1")
    for col in df.columns:  # some numeric answers are stored as text
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _district_code(folder, name="lahore") -> int:
    _, meta = pyreadstat.read_dta(f"{folder}/info.dta", metadataonly=True, encoding="latin1")
    labels = meta.value_labels[meta.variable_to_label["district"]]
    return next(code for code, label in labels.items() if label == name)


def _household_income(folder, hh_ids) -> pd.Series:
    cols = ["hhcode", "seaq01", "seaq07", "seaq08", "seaq09", "seaq10", "seaq15", "seaq17",
            "seaq19", "seaq21", "seaq23", "seaq24", "seaq25", "seaq26"]
    e = _read(folder, "sece", cols)
    e = e[e["hhcode"].isin(hh_ids)].copy()
    monthly_basis = e["seaq08"].fillna(0) * e["seaq09"].fillna(12)
    annual_basis = e["seaq10"].fillna(0)
    main_job = np.select([e["seaq07"] == 1, e["seaq07"] == 2], [monthly_basis, annual_basis],
                         default=np.where(e["seaq10"].notna(), annual_basis,
                                          np.where(e["seaq08"].notna(), monthly_basis, 0.0)))
    other = ["seaq15", "seaq17", "seaq19", "seaq21", "seaq23", "seaq24", "seaq25", "seaq26"]
    e["annual"] = main_job + e[other].fillna(0).sum(axis=1)
    e["worked"] = e["seaq01"] == 1
    e["answered"] = e[["seaq23", "seaq24", "seaq25", "seaq26"]].notna().any(axis=1)
    g = e.groupby("hhcode")
    total, worked, answered = g["annual"].sum(), g["worked"].any(), g["answered"].any()
    measured = (total > 0) | (~worked & answered)
    return (total.where(measured) / 12.0).rename("monthly_income")


def build_lahore_households(zip_path=C.PSLM_ZIP) -> tuple[pd.DataFrame, dict]:
    folder = _extract(zip_path, C.RAW_DIR)
    lahore = _district_code(folder)

    roster = _read(folder, "roster", ["hhcode", "idc", "district", "age", "sb1q2", "sb1q11"])
    members = roster[(roster["district"] == lahore) & (roster["sb1q11"] == 1)]
    educ = _read(folder, "secc1", ["hhcode", "idc", "sc1q01", "sc1q1a", "sc1q2a", "sc1q3a", "sc1q18"])
    members = members.merge(educ, on=["hhcode", "idc"], how="left")

    kids = members[members["age"].between(*C.SCHOOL_AGE) & members["sc1q01"].notna()]
    g = kids.groupby("hhcode")["sc1q01"]
    hh = pd.DataFrame({C.TARGET_COLUMN: g.apply(lambda s: (s == 3).mean()),
                       "n_school_age": g.size()})

    head = members[members["sb1q2"] == 1].drop_duplicates("hhcode").set_index("hhcode")
    hh["read_write"] = ((head["sc1q1a"] == 1) & (head["sc1q2a"] == 1)).astype(int).reindex(hh.index)
    hh["solve_math"] = (head["sc1q3a"] == 1).astype(int).reindex(hh.index)
    hh = hh.join(_household_income(folder, set(hh.index)))

    # sc1q18 "how far (round trip) is the institution from home": asked only of
    # children who attend; the release carries no value labels for its codes.
    attending = kids[kids["sc1q01"] == 3]
    log = {"attending_children_by_distance_code_sc1q18": {
               str(int(k)): round(float(v), 3)
               for k, v in attending["sc1q18"].value_counts(normalize=True).sort_index().items()},
           "lahore_households_with_school_age_children": len(hh),
           "dropped_income_unmeasured_or_missing": int(hh["monthly_income"].isna().sum()),
           "dropped_no_identified_head": int(hh["read_write"].isna().sum())}
    hh = hh.dropna(subset=["monthly_income", "read_write", "solve_math"]).reset_index(drop=True)
    hh[["read_write", "solve_math"]] = hh[["read_write", "solve_math"]].astype(int)
    log.update({"households_used": len(hh),
                "school_age_children": int(hh["n_school_age"].sum()),
                "child_attendance_rate": round(float(np.average(hh[C.TARGET_COLUMN], weights=hh["n_school_age"])), 3),
                "households_enrolled_share_ge_0.5": round(float((hh[C.TARGET_COLUMN] >= C.ENROLLED_THRESHOLD).mean()), 3),
                # household monthly income at the 20th, 40th, 60th, 80th percentiles (PKR)
                "income_quintile_cutpoints": [round(float(q)) for q in
                                              hh["monthly_income"].quantile([0.2, 0.4, 0.6, 0.8])]})
    return hh, log
