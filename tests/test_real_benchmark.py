"""Functional tests for the real-data benchmark: anti-overfit, control, explainability."""

from __future__ import annotations

from pipelineguard.features import FEATURE_COLUMNS
from pipelineguard.real_benchmark import format_report, main


def test_clears_the_baseline_in_and_out_of_sample(report):
    assert report.cv_auc_mean > 0.85
    assert report.holdout_auc > 0.85
    assert report.holdout_accuracy > report.majority_accuracy
    assert report.beats_baseline


def test_is_well_fit_not_overfit(report):
    # Train AUC runs ahead of held-out (some generalisation cost), but only a little.
    assert report.train_auc > report.holdout_auc
    assert report.gap <= 0.08, report
    assert report.verdict.startswith("well-fit"), report.verdict


def test_cross_val_agrees_with_the_held_out_test(report):
    # Stability: the cross-validated AUC and the single held-out AUC land together.
    assert abs(report.cv_auc_mean - report.holdout_auc) < 0.03


def test_shuffled_label_control_collapses_to_chance(report):
    # Permuting the labels destroys the signal: the control must fall to ~chance...
    assert report.shuffle_auc < 0.60
    # ...and sit far below the real detector. That gap is the learned signal.
    assert report.holdout_auc - report.shuffle_auc > 0.25


def test_regression_cross_check_finds_signal(report):
    # The signal is also there on the continuous corrosion rate, not just the threshold.
    assert report.regression_r2 > 0.4


def test_report_exposes_full_explainability(report):
    assert len(report.importances) == len(FEATURE_COLUMNS)
    assert len(report.coefficients) == len(FEATURE_COLUMNS)
    assert report.importances[0].feature in FEATURE_COLUMNS
    assert report.importances[0].importance > 0.05


def test_format_report_renders_the_evidence(report):
    text = format_report(report)
    assert "wellhead integrity" in text
    assert "majority baseline" in text
    assert "shuffled-label" in text
    assert f"{report.holdout_auc:.4f}" in text
    assert "well-fit" in text


def test_main_prints_the_report(capsys):
    main()
    assert "real-data benchmark" in capsys.readouterr().out
