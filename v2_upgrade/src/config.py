from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = V2_ROOT / "data" / "raw"
DATA_PROCESSED = V2_ROOT / "data" / "processed"

# aliases (some scripts import these names)
PROCESSED_DIR = DATA_PROCESSED
RAW_DIR = DATA_RAW

MODELS_DIR = V2_ROOT / "models"
ARTIFACTS_DIR = V2_ROOT / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

for p in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)

DEFAULT_MARGIN_PCT = 0.6
DEFAULT_ANNUAL_DISCOUNT = 0.1
DEFAULT_HORIZON_MONTHS = 24
