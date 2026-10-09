"""Functional tests for the integrity detector."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from pipelineguard.features import FEATURE_COLUMNS, build_features, split_xy
from pipelineguard.model import IntegrityDetector


@pytest.fixture(scope="module")
def split(wellheads):
    x, y = split_xy(build_features(wellheads))
    return train_test_split(x, np.asarray(y), test_size=0.2, random_state=42, stratify=y)


@pytest.fixture(scope="module")
def fitted(split):
    x_tr, _x_te, y_tr, _y_te = split
    return IntegrityDetector().fit(x_tr, y_tr)


def test_beats_the_majority_baseline_on_holdout(split, fitted):
    _x_tr, x_te, _y_tr, y_te = split
    metrics = fitted.evaluate(x_te, y_te)
    majority = 1 - y_te.mean()  # ~0.75
    # A useful detector clears the "always say normal" floor by a wide margin.
    assert metrics.roc_auc > 0.85
    assert metrics.accuracy > majority
    assert metrics.f1 > 0.6


def test_feature_importances_exposed_and_nontrivial(split, fitted):
    _x_tr, x_te, _y_tr, y_te = split
    imp = fitted.feature_importances(x_te, y_te, n_repeats=5)
    assert len(imp) == len(FEATURE_COLUMNS)
    # Sorted high to low, and the leading feature genuinely matters.
    assert imp == sorted(imp, key=lambda fi: fi.importance, reverse=True)
    assert imp[0].importance > 0.05
    # A physics-aware feature is among the real drivers.
    top5 = {fi.feature for fi in imp[:5]}
    assert "co2_partial_pressure_psi" in top5


def test_linear_coefficients_are_signed_and_readable(fitted):
    coefs = fitted.linear_coefficients()
    assert len(coefs) == len(FEATURE_COLUMNS)
    assert coefs == sorted(coefs, key=lambda c: abs(c.coefficient), reverse=True)
    assert abs(coefs[0].coefficient) > 0.1  # a non-trivial leading driver


def test_probabilities_and_hard_flags(split, fitted):
    _x_tr, x_te, _y_tr, _y_te = split
    proba = fitted.predict_proba(x_te)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    flags = fitted.predict(x_te)
    assert set(np.unique(flags)).issubset({0, 1})


def test_unfitted_detector_refuses_to_predict():
    with pytest.raises(RuntimeError):
        IntegrityDetector().predict_proba(np.zeros((1, len(FEATURE_COLUMNS))))
