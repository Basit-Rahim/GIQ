"""
Step 2 - Build the PSLM 2019-20 Lahore validation table from the PBS download.

    python scripts/build_pslm.py

Reads  data/raw/stata_data_pslm_2019_20.zip
Writes data/interim/pslm_lahore_households.csv   (not committed: PBS microdata)
       outputs/pslm_sample.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src.pslm import build_lahore_households


def main():
    if not C.PSLM_ZIP.exists():
        sys.exit(f"Missing {C.PSLM_ZIP.relative_to(C.ROOT)}. Download the PSLM 2019-20 "
                 "microdata (Stata format) from the Pakistan Bureau of Statistics; see README.")
    households, log = build_lahore_households(C.PSLM_ZIP)
    C.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    C.OUTPUTS_DIR.mkdir(exist_ok=True)
    households.to_csv(C.PSLM_TABLE, index=False)
    (C.OUTPUTS_DIR / "pslm_sample.json").write_text(json.dumps(log, indent=2))
    print(f"{len(households)} households -> {C.PSLM_TABLE.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
