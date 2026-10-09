"""Real wellhead process data, and how the integrity target is derived from it.

PipelineGuard is built and validated on **real** wellhead measurements, not a toy
generator: the ``maisaian/oil-gas-leakage`` dataset on Hugging Face (10,292 rows of
nine float columns from oil & gas wellheads). Each row is one well's steady-state
process snapshot — temperature, pressure, gas/oil/water rates, basic sediment & water,
CO2 content, gas gravity — paired with a measured ``CR-corrosion defect`` reading.

**The target (and why).** ``CR-corrosion defect`` is the integrity signal: the measured
internal corrosion rate at the wellhead. Rather than regress its exact value (noisy in
the long right tail, and operationally the question is binary — *do we flag this well for
inspection or not?*), PipelineGuard frames it as **detection**: a well is ``high_risk``
when its corrosion rate sits in the worst quartile of the fleet. That threshold (the 75th
percentile of ``CR-corrosion defect``) is a fixed operating definition applied identically
to every row; it turns a continuous reading into the inspect / do-not-inspect decision a
reliability team actually makes, and gives the ROC-AUC / F1 vocabulary that goes with it.

**No leakage.** ``CR-corrosion defect`` is used *only* to build the label. It is never a
feature, and no feature is a transform of it (:mod:`pipelineguard.features` builds every
input purely from the process columns). The corrosion reading that defines the target
therefore never leaks into the inputs that predict it.

The default demand is the vendored slice under ``datasets/wellhead_integrity_real.csv``;
``fetch_wellhead_data`` refreshes it from the Hugging Face datasets-server.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPO_ROOT / "datasets" / "wellhead_integrity_real.csv"

#: Hugging Face dataset id (url-encoded) and the datasets-server rows endpoint.
_DATASET_ID = "maisaian%2Foil-gas-leakage"
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={ds}"
    "&config=default&split=train&offset={off}&length=100"
)

#: Original Hugging Face column names, in file order.
SOURCE_COLUMNS = [
    "Wellhead Temp. (C)",
    "Wellhead Press (psi)",
    "MMCFD- gas",
    "BOPD (barrel of oil produced per day)",
    "BWPD (barrel of water produced per day)",
    "BSW - basic solid and water (%)",
    "CO2 mol. (%) @ 25 C & 1 Atm.",
    "Gas Grav.",
    "CR-corrosion defect",
]

#: Map the raw source names onto the tidy snake_case names the package uses.
_RENAME = {
    "Wellhead Temp. (C)": "wellhead_temp_c",
    "Wellhead Press (psi)": "wellhead_press_psi",
    "MMCFD- gas": "gas_rate_mmcfd",
    "BOPD (barrel of oil produced per day)": "oil_rate_bopd",
    "BWPD (barrel of water produced per day)": "water_rate_bwpd",
    "BSW - basic solid and water (%)": "bsw_pct",
    "CO2 mol. (%) @ 25 C & 1 Atm.": "co2_mol_pct",
    "Gas Grav.": "gas_gravity",
    "CR-corrosion defect": "corrosion_rate",
}

#: The raw process-sensor columns (everything except the corrosion reading).
RAW_COLUMNS = [v for k, v in _RENAME.items() if k != "CR-corrosion defect"]

#: The corrosion reading that *defines* the target. Never used as a feature.
CORROSION_COLUMN = "corrosion_rate"

#: Binary target column added by :func:`derive_target`.
TARGET = "high_risk"

#: A well is flagged high-risk when its corrosion rate is in this top fraction of the
#: fleet. The 75th percentile is the fixed operating definition (worst quartile).
HIGH_RISK_QUANTILE = 0.75


def load_wellhead_data(with_corrosion: bool = False) -> pd.DataFrame:
    """Load the vendored real wellhead data with the derived integrity target.

    Returns a frame with the ten tidy process columns in :data:`RAW_COLUMNS` plus the
    binary :data:`TARGET` (``high_risk``). The raw ``corrosion_rate`` reading that
    defines the target is dropped unless ``with_corrosion`` is set (the benchmark keeps
    it to compute the threshold and a regression cross-check); it must never be fed to
    the model as a feature.
    """
    if not _DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Vendored wellhead data not found at {_DATASET_PATH}. Run "
            "`python -c 'from pipelineguard.real_data import fetch_wellhead_data; "
            "fetch_wellhead_data()'`."
        )
    df = pd.read_csv(_DATASET_PATH)
    df = df.rename(columns=_RENAME)
    df = derive_target(df)
    cols = [*RAW_COLUMNS, TARGET]
    if with_corrosion:
        cols = [*RAW_COLUMNS, CORROSION_COLUMN, TARGET]
    return df[cols].reset_index(drop=True)


def risk_threshold(corrosion_rate: pd.Series) -> float:
    """The corrosion-rate value that separates high-risk from normal wells.

    Defined as the :data:`HIGH_RISK_QUANTILE` quantile of the fleet's corrosion
    readings, so it is reproducible and fleet-relative rather than an arbitrary number.
    """
    return float(corrosion_rate.quantile(HIGH_RISK_QUANTILE))


def derive_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary :data:`TARGET` by thresholding the corrosion-rate reading.

    ``high_risk = 1`` when ``corrosion_rate`` is at or above the fleet threshold
    (:func:`risk_threshold`), else ``0``. This is the only use of the corrosion column.
    """
    out = df.copy()
    threshold = risk_threshold(out[CORROSION_COLUMN])
    out[TARGET] = (out[CORROSION_COLUMN] >= threshold).astype(int)
    return out


def _get(off: int, tries: int = 5) -> dict:
    """Fetch one 100-row page from the datasets-server, with exponential-backoff retries.

    Isolated so tests can monkeypatch the network away.
    """
    url = _ROWS_URL.format(ds=_DATASET_ID, off=off)
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - fixed https host
                return json.load(resp)
        except Exception:  # noqa: BLE001 - network retry
            if i == tries - 1:
                raise
            time.sleep(2**i)
    return {}


def fetch_wellhead_data(dest: str | Path | None = None, max_rows: int = 11000) -> int:
    """Download the full wellhead dataset into the vendored CSV, with original columns.

    Pages the Hugging Face datasets-server 100 rows at a time until the split is
    exhausted (or ``max_rows`` is reached). Writes the raw source column names so the
    vendored file stays a faithful copy of the upstream data. Returns the row count.
    """
    target = Path(dest) if dest is not None else _DATASET_PATH
    rows: list[dict] = []
    off = 0
    while off < max_rows:
        page = _get(off).get("rows", [])
        if not page:
            break
        rows.extend(r["row"] for r in page)
        off += 100
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SOURCE_COLUMNS)
        for r in rows:
            writer.writerow([r[c] for c in SOURCE_COLUMNS])
    return len(rows)
