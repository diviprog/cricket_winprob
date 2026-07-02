"""Project-wide constants and the single train/held-out split definition.

Every modeling decision that spec 07 says must be recorded lives here or is
documented at its point of use in the relevant module.
"""

from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
BALLS_PARQUET = DATA_PROCESSED / "second_innings_balls.parquet"
RAW_CSV = DATA_RAW / "IPL.csv"

# --- T20 structural constants ----------------------------------------------
BALLS_PER_INNINGS = 120  # 20 overs * 6 legal balls
WICKETS = 10

# --- outcome categories (spec 01) ------------------------------------------
OUTCOMES = ["0", "1", "2", "3", "4", "5", "6", "W"]

# --- ties (spec 01 decision: keep ties, terminal value 0.5) ----------------
TIE_VALUE = 0.5

# --- train / held-out split (spec 02/05/06: hold out most recent 3 seasons)
# Defined ONCE here and reused everywhere. The concrete season labels are
# resolved from the data at ingestion time (seasons are ordered by first match
# date) and written into the parquet as a `split` column, so the split is fixed
# on disk and never recomputed inconsistently downstream.
N_HELDOUT_SEASONS = 3
