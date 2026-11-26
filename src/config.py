"""Global configuration for the MIT-BIH arrhythmia beat classification project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mitdb"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

for _d in (DATA_DIR, OUT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FS = 360.0                 # MIT-BIH sampling frequency (Hz)
LEAD_PREFERENCE = ["MLII", "II", "V5", "V1"]

# Records excluded from the study, following AAMI EC57 practice: records 102, 104,
# 107 and 217 are dominated by paced rhythms, and the paced class is not part of
# the AAMI super-class taxonomy used here. de Chazal et al. (2004) exclude the
# same four records, which is why their DS1/DS2 split covers 44 records.
EXCLUDED_RECORDS = [102, 104, 107, 217]

# The 44 analysed records of the MIT-BIH Arrhythmia Database (= DS1 + DS2).
# Inter-patient split proposed by de Chazal et al. (2004); the de-facto standard
# protocol for beat-level evaluation on MIT-BIH.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 117, 123, 200, 202, 210, 212, 213, 214,
       219, 221, 222, 228, 231, 232, 233, 234]

RECORDS = sorted(DS1 + DS2)

# AAMI EC57 super-classes.
AAMI_MAP = {
    # normal / bundle-branch block
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # ventricular ectopic
    "V": "V", "E": "V",
    # fusion
    "F": "F",
    # unknown / paced - not represented in the 44 analysed records
    "/": "Q", "f": "Q", "Q": "Q",
}
# Only the four classes that actually occur in the analysed records are modelled;
# the paced class Q lives exclusively in the four excluded paced recordings.
CLASSES = ["N", "S", "V", "F"]
CLASS_NAMES = {
    "N": "Normal (N)",
    "S": "Supraventricular ectopic (S)",
    "V": "Ventricular ectopic (V)",
    "F": "Fusion (F)",
    "Q": "Unknown / paced (Q)",
}

# Beat window (seconds relative to the R peak).
PRE_S = 0.25
POST_S = 0.45
PRE = int(round(PRE_S * FS))     # 90 samples
POST = int(round(POST_S * FS))   # 162 samples
WIN = PRE + POST                 # 252 samples

# Detection tolerance used when matching Pan-Tompkins detections to annotations.
MATCH_TOL_S = 0.150

RANDOM_STATE = 42
