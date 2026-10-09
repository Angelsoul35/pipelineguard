"""Functional test for the CLI demo."""

from __future__ import annotations

import pipelineguard.cli as cli


def test_demo_runs_end_to_end(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["pipelineguard", "demo", "--repeats", "3"])
    cli.main()
    out = capsys.readouterr().out
    # The demo walks the whole story: data -> train -> explain -> control.
    assert "Loading real wellhead data" in out
    assert "held-out ROC-AUC" in out
    assert "shuffled-label control" in out.lower() or "Shuffled-label control" in out
    # It prints at least one named driver among the top features.
    assert "wellhead_press_psi" in out or "co2_partial_pressure_psi" in out
