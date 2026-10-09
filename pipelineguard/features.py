"""Turn raw wellhead readings into a physics-aware feature matrix.

The model does not see the nine raw numbers directly. It sees a representation built
around the chemistry and flow physics of CO2 ("sweet") corrosion — the dominant internal
corrosion mechanism at an oil & gas wellhead. Each engineered feature is a quantity a
corrosion/reliability engineer would compute by hand:

* **CO2 partial pressure** ``pCO2 = (CO2 mol% / 100) x wellhead pressure`` (psi). CO2
  does not corrode steel as a gas; it dissolves into produced water and forms carbonic
  acid (``CO2 + H2O -> H2CO3``), which attacks the pipe wall. The *partial pressure* of
  CO2 — not the mole fraction alone — is what sets how much dissolves (Henry's law), and
  it is the primary driver in the de Waard-Milliams corrosion-rate correlations. Encoding
  CO2 this way folds the raw mole-% and pressure into the one physically meaningful term.
* **Water cut** ``water_cut = BWPD / (BOPD + BWPD)``. The fraction of produced liquid
  that is water. Water is the electrolyte: no continuous water phase at the wall, little
  aqueous corrosion. A rising water cut is a classic integrity red flag.
* **Total liquid** ``total_liquid = BOPD + BWPD`` (bbl/day). Throughput. Together with
  water cut this is just a physically-meaningful reparametrization of the raw oil and
  water rates (``BOPD = total_liquid x (1 - water_cut)``), into the two axes — how much
  fluid, and how wet it is — that govern flow regime and wall wetting.
* **Gas-oil ratio** ``GOR = gas_rate x 1e6 / BOPD`` (scf/bbl). High GOR shifts the flow
  regime toward mist/annular flow and raises wall shear, which can strip protective
  FeCO3 scale and drive erosion-corrosion.
* **Water-balance residual** ``residual = BSW/100 - water_cut``. A **mass-balance
  consistency check**: BSW (basic sediment & water) and the flow-derived water cut are
  two independent estimates of the same water fraction, so they should agree. A large
  residual means the water accounting does not close — a meter drift or separation
  anomaly, exactly the kind of process upset that precedes an integrity event — so the
  residual is carried as its own feature/flag.

Wellhead temperature (reaction kinetics), total operating pressure (flow regime and
in-situ conditions, an effect beyond the pCO2 term), gas rate and gas gravity (gas-phase
composition) round out the set. Crucially, **no feature touches the corrosion reading**
that defines the target, so there is no label leakage.
"""

from __future__ import annotations

import pandas as pd

from .real_data import TARGET

#: Raw process sensors kept as-is (they carry signal the engineered terms do not fully
#: capture: e.g. operating pressure acts on corrosion through several channels, not only
#: through the CO2 partial-pressure term).
PASSTHROUGH_FEATURES = [
    "wellhead_temp_c",
    "wellhead_press_psi",
    "gas_rate_mmcfd",
    "gas_gravity",
    "bsw_pct",
]

#: Physics-aware features added by :func:`build_features`.
DERIVED_FEATURES = [
    "co2_partial_pressure_psi",
    "water_cut",
    "total_liquid_bpd",
    "gas_oil_ratio",
    "water_balance_residual",
]

#: The exact model input order (passthrough + derived, interleaved for readability).
FEATURE_COLUMNS = [
    "wellhead_temp_c",
    "wellhead_press_psi",
    "gas_rate_mmcfd",
    "gas_gravity",
    "co2_partial_pressure_psi",
    "water_cut",
    "total_liquid_bpd",
    "gas_oil_ratio",
    "bsw_pct",
    "water_balance_residual",
]


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the five physics-aware derived columns added.

    Every formula uses only process-sensor columns; none uses the corrosion reading.
    """
    out = df.copy()
    out["co2_partial_pressure_psi"] = (out["co2_mol_pct"] / 100.0) * out["wellhead_press_psi"]
    total_liquid = out["oil_rate_bopd"] + out["water_rate_bwpd"]
    out["total_liquid_bpd"] = total_liquid
    out["water_cut"] = out["water_rate_bwpd"] / total_liquid
    out["gas_oil_ratio"] = out["gas_rate_mmcfd"] * 1_000_000.0 / out["oil_rate_bopd"]
    out["water_balance_residual"] = (out["bsw_pct"] / 100.0) - out["water_cut"]
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the physics features and return a frame carrying the full model input set."""
    return add_physics_features(df)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature frame into the model inputs ``X`` and the binary target ``y``.

    ``X`` is exactly :data:`FEATURE_COLUMNS` — the corrosion reading is never included.
    """
    return df[FEATURE_COLUMNS], df[TARGET]
