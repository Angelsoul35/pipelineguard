"""PipelineGuard command-line interface."""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from .features import build_features, split_xy
from .model import IntegrityDetector
from .real_data import load_wellhead_data


def cmd_demo(args) -> None:
    print("1) Loading real wellhead data ...")
    df = load_wellhead_data()
    print(f"   {len(df)} wells  |  high-risk base rate {df['high_risk'].mean():.1%}  "
          f"|  majority baseline acc {1 - df['high_risk'].mean():.1%}")

    print("2) Training the physics-aware detector ...")
    x, y = split_xy(build_features(df))
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    detector = IntegrityDetector().fit(x_tr, y_tr)
    metrics = detector.evaluate(x_te, y_te)
    print(f"   held-out ROC-AUC {metrics.roc_auc}  F1 {metrics.f1}  acc {metrics.accuracy}")

    print("3) What drives the risk (top features, held-out permutation importance) ...")
    for fi in detector.feature_importances(x_te, y_te, n_repeats=args.repeats)[:4]:
        print(f"   {fi.feature:26s} {fi.importance:+.4f}")

    print("4) Shuffled-label control (should collapse to chance ~0.5) ...")
    rng = np.random.default_rng(0)
    control = IntegrityDetector().fit(x_tr, rng.permutation(np.asarray(y_tr)))
    print(f"   control held-out ROC-AUC {control.evaluate(x_te, y_te).roc_auc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PipelineGuard CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="train the detector on real data and explain it")
    p.add_argument("--repeats", type=int, default=10, help="permutation-importance repeats")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
