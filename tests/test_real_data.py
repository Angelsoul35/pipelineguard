"""Functional tests for the real-data loader, target derivation, and fetch path."""

from __future__ import annotations

import pytest

import pipelineguard.real_data as rd
from pipelineguard.real_data import (
    CORROSION_COLUMN,
    RAW_COLUMNS,
    TARGET,
    derive_target,
    load_wellhead_data,
    risk_threshold,
)


def test_loads_the_real_fleet(wellheads):
    # The upstream maisaian/oil-gas-leakage split has exactly 10,292 wells.
    assert len(wellheads) == 10292
    # Physically sane ranges for real wellhead readings.
    assert 40 < wellheads["wellhead_temp_c"].min() and wellheads["wellhead_temp_c"].max() < 80
    assert 300 < wellheads["wellhead_press_psi"].min()
    assert wellheads["wellhead_press_psi"].max() < 2500
    assert (wellheads["oil_rate_bopd"] > 0).all()
    assert (wellheads["water_rate_bwpd"] > 0).all()
    assert 0 <= wellheads["bsw_pct"].min() and wellheads["bsw_pct"].max() <= 100


def test_default_load_hides_the_corrosion_reading():
    df = load_wellhead_data()
    assert list(df.columns) == [*RAW_COLUMNS, TARGET]
    assert CORROSION_COLUMN not in df.columns  # the label source must not ride along


def test_with_corrosion_keeps_the_reading(wellheads):
    assert CORROSION_COLUMN in wellheads.columns
    assert list(wellheads.columns) == [*RAW_COLUMNS, CORROSION_COLUMN, TARGET]


def test_target_is_the_worst_quartile(wellheads):
    # high_risk is the top 25% of corrosion rate: base rate must be ~0.25.
    assert wellheads[TARGET].mean() == pytest.approx(0.25, abs=0.01)
    thr = risk_threshold(wellheads[CORROSION_COLUMN])
    assert thr == pytest.approx(0.2328, abs=1e-3)
    # The label is exactly corrosion_rate >= threshold (the only use of the reading).
    expected = (wellheads[CORROSION_COLUMN] >= thr).astype(int)
    assert (wellheads[TARGET] == expected).all()


def test_derive_target_is_deterministic():
    df = load_wellhead_data(with_corrosion=True)
    again = derive_target(df[[CORROSION_COLUMN]])
    assert (again[TARGET] == df[TARGET]).all()


def test_loader_rejects_a_missing_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(rd, "_DATASET_PATH", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        load_wellhead_data()


def test_fetch_pages_and_writes_original_columns(tmp_path, monkeypatch):
    page = [
        {"row": {
            "Wellhead Temp. (C)": 53.35, "Wellhead Press (psi)": 1105.13,
            "MMCFD- gas": 12.87, "BOPD (barrel of oil produced per day)": 1378.93,
            "BWPD (barrel of water produced per day)": 2812.62,
            "BSW - basic solid and water (%)": 75.64,
            "CO2 mol. (%) @ 25 C & 1 Atm.": 3.3628, "Gas Grav.": 0.7205,
            "CR-corrosion defect": 0.2245,
        }},
        {"row": {
            "Wellhead Temp. (C)": 72.25, "Wellhead Press (psi)": 1026.31,
            "MMCFD- gas": 3.42, "BOPD (barrel of oil produced per day)": 1028.75,
            "BWPD (barrel of water produced per day)": 919.92,
            "BSW - basic solid and water (%)": 44.21,
            "CO2 mol. (%) @ 25 C & 1 Atm.": 3.8679, "Gas Grav.": 0.894,
            "CR-corrosion defect": 0.2262,
        }},
    ]
    calls = {"n": 0}

    def fake_get(off, tries=5):
        calls["n"] += 1
        return {"rows": page if off == 0 else []}  # one page, then exhausted

    monkeypatch.setattr(rd, "_get", fake_get)
    dest = tmp_path / "fetched.csv"
    n = rd.fetch_wellhead_data(dest=dest)
    assert n == 2
    assert calls["n"] >= 2  # fetched page 0, then the empty page that stops paging

    # The written file is a faithful copy that the loader can read back.
    monkeypatch.setattr(rd, "_DATASET_PATH", dest)
    out = load_wellhead_data(with_corrosion=True)
    assert out["wellhead_temp_c"].tolist() == [53.35, 72.25]
    assert out["co2_mol_pct"].tolist() == [3.3628, 3.8679]
    assert out[CORROSION_COLUMN].tolist() == [0.2245, 0.2262]


def test_get_retries_then_raises(monkeypatch):
    attempts = {"n": 0}

    def boom(url, timeout=60):
        attempts["n"] += 1
        raise OSError("network down")

    monkeypatch.setattr(rd.urllib.request, "urlopen", boom)
    monkeypatch.setattr(rd.time, "sleep", lambda *_: None)  # no real backoff waiting
    with pytest.raises(OSError):
        rd._get(0, tries=3)
    assert attempts["n"] == 3  # retried the full budget before giving up
