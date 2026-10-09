"""Functional tests for the physics-aware features (and the no-leakage guard)."""

from __future__ import annotations

import pytest

from pipelineguard.features import (
    DERIVED_FEATURES,
    FEATURE_COLUMNS,
    add_physics_features,
    build_features,
    split_xy,
)
from pipelineguard.real_data import CORROSION_COLUMN, TARGET


def test_physics_features_pinned_by_hand(wellheads):
    """First well: every derived feature recomputed by hand from the raw readings.

    Row 0 raw: press 1105.13 psi, CO2 3.3628 mol%, oil 1378.93 BOPD, water 2812.62 BWPD,
    gas 12.87 MMCFD, BSW 75.64 %.
    """
    feat = add_physics_features(wellheads)
    row = feat.iloc[0]
    # pCO2 = (3.3628 / 100) * 1105.13 = 37.163312 psi
    assert row["co2_partial_pressure_psi"] == pytest.approx(37.163312, abs=1e-4)
    # water cut = 2812.62 / (1378.93 + 2812.62) = 0.671021
    assert row["water_cut"] == pytest.approx(0.671021, abs=1e-6)
    # total liquid = 1378.93 + 2812.62 = 4191.55 bbl/day
    assert row["total_liquid_bpd"] == pytest.approx(4191.55, abs=1e-6)
    # GOR = 12.87e6 / 1378.93 = 9333.3237 scf/bbl
    assert row["gas_oil_ratio"] == pytest.approx(9333.3237, abs=1e-3)
    # water-balance residual = 75.64/100 - 0.671021 = 0.085379
    assert row["water_balance_residual"] == pytest.approx(0.085379, abs=1e-5)


def test_water_cut_is_a_fraction(wellheads):
    feat = add_physics_features(wellheads)
    assert feat["water_cut"].between(0, 1).all()


def test_feature_set_excludes_the_label(wellheads):
    """The leakage guard: neither the corrosion reading nor the target is ever a feature."""
    assert CORROSION_COLUMN not in FEATURE_COLUMNS
    assert TARGET not in FEATURE_COLUMNS
    x, y = split_xy(build_features(wellheads))
    assert CORROSION_COLUMN not in x.columns
    assert TARGET not in x.columns
    assert list(x.columns) == FEATURE_COLUMNS
    assert y.name == TARGET


def test_all_derived_features_present(wellheads):
    feat = build_features(wellheads)
    for name in DERIVED_FEATURES:
        assert name in feat.columns
    assert len(FEATURE_COLUMNS) == 10
