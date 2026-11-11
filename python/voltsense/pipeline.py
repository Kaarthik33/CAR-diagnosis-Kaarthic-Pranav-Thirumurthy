"""End-to-end run: simulate fleet -> C++ features -> ML -> report.

    python -m voltsense.pipeline               # full run (60 vehicles x 60 days)
    python -m voltsense.pipeline --quick       # CI smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import REPO_ROOT, add_fleet_features, extract_features, label_trips
from .model import (
    CLASSES,
    alarm_threshold,
    cross_validate,
    diagnosis_confusion,
    feature_sets,
    lead_time_report,
)
from .simulator import FleetConfig, simulate_fleet

COLORS = {"none": "#7a869a", "healthy": "#7a869a", "battery": "#1f6feb", "bearing": "#d97706", "cooling": "#c2185b"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


def plot_degradation(df: pd.DataFrame, out: Path) -> None:
    panels = [("r_int_ohm_roll", "Internal resistance R_int [Ω]", "battery"),
              ("thermal_conductance_roll", "Motor thermal conductance b [1/min]", "cooling"),
              ("vib_floor_roll", "Drive-unit vibration floor [g]", "bearing")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (col, title, fault) in zip(axes, panels):
        healthy = df[df.fault == "none"].vehicle_id.unique()[:6]
        faulty = df[df.fault == fault].vehicle_id.unique()[:3]
        for v in healthy:
            g = df[df.vehicle_id == v]
            ax.plot(g.day, g[col], color=COLORS["none"], lw=0.8, alpha=0.6)
        for v in faulty:
            g = df[df.vehicle_id == v]
            ax.plot(g.day, g[col], color=COLORS[fault], lw=1.6)
            ax.axvline(g.failure_day.iloc[0], color=COLORS[fault], ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("day")
    axes[0].plot([], [], color=COLORS["none"], label="healthy vehicles")
    axes[0].plot([], [], color=COLORS["battery"], label="degrading vehicle (dotted = failure)")
    axes[0].legend(loc="center left", frameon=False, fontsize=7)
    fig.suptitle("Physical parameters identified on-line by the C++ RLS estimators", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_ablation(results, out: Path) -> None:
    names = [r.name for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, attr, title in [(axes[0], "macro_f1", "Root-cause diagnosis (macro-F1)"),
                            (axes[1], "alarm_pr_auc", "Failure-within-7-days alarm (PR-AUC)")]:
        vals = [getattr(r, attr) for r in results]
        errs = [getattr(r, attr + "_std") for r in results]
        bars = ax.barh(names, vals, xerr=errs, color=["#b8c2d1", "#6c8ebf", "#1f6feb"], capsize=3)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=6, fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.invert_yaxis()
        ax.set_title(title)
    fig.suptitle("Ablation - 5-fold cross-validation grouped by vehicle (mean ± std)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out: Path) -> None:
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (fails within 7 days)")
    ax.set_title("Diagnosis confusion matrix (out-of-fold)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_lead_time(lead: pd.DataFrame, out: Path) -> None:
    faulty = lead[lead.fault != "none"]
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    for f in ["battery", "bearing", "cooling"]:
        v = faulty[faulty.fault == f].lead_days.dropna()
        ax.scatter(v, np.full(len(v), f), color=COLORS[f], s=28, alpha=0.8)
    missed = faulty.lead_days.isna().sum()
    ax.axvline(7, color="black", ls=":", lw=0.8)
    ax.text(7.15, 1.5, "7-day\ntarget", fontsize=7)
    ax.set_xlabel("days of warning before failure")
    ax.set_title(f"Warning lead time per failing vehicle (missed: {missed})")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small fleet for CI")
    ap.add_argument("--vehicles", type=int, default=60)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs")
    args = ap.parse_args(argv)

    cfg = FleetConfig(n_vehicles=20 if args.quick else args.vehicles,
                      n_days=40 if args.quick else args.days, seed=args.seed)
    timings = {}

    t = time.perf_counter()
    telemetry, vehicles = simulate_fleet(cfg)
    timings["simulate_s"] = time.perf_counter() - t
    print(f"[1/4] simulated {len(telemetry):,} samples from {cfg.n_vehicles} vehicles")

    t = time.perf_counter()
    trips = extract_features(telemetry)
    timings["extract_s"] = time.perf_counter() - t
    print(f"[2/4] extracted features for {len(trips):,} trips in {timings['extract_s']:.2f}s")

    df = label_trips(add_fleet_features(trips), vehicles)

    t = time.perf_counter()
    results = [cross_validate(df, cols, name) for name, cols in feature_sets(df).items()]
    timings["train_eval_s"] = time.perf_counter() - t
    for r in results:
        print(f"[3/4] {r.name:<30} macro-F1 {r.macro_f1:.3f} ± {r.macro_f1_std:.3f}   "
              f"alarm PR-AUC {r.alarm_pr_auc:.3f} ± {r.alarm_pr_auc_std:.3f}")

    best = results[-1]
    risk = 1.0 - best.oof_proba[:, 0]
    thr = alarm_threshold(df, risk, max_false_alarm_rate=0.01)
    lead = lead_time_report(df, risk, thr)
    faulty = lead[lead.fault != "none"]
    healthy = lead[lead.fault == "none"]
    detected = faulty.lead_days.notna() & (faulty.lead_days >= 1)
    cm = diagnosis_confusion(df, best.oof_proba)

    summary = {
        "fleet": {"vehicles": cfg.n_vehicles, "days": cfg.n_days, "samples": len(telemetry),
                  "trips": len(df), "failing_vehicles": len(faulty)},
        "ablation": [{"features": r.name, "macro_f1": round(r.macro_f1, 3), "macro_f1_std": round(r.macro_f1_std, 3),
                      "alarm_pr_auc": round(r.alarm_pr_auc, 3), "alarm_pr_auc_std": round(r.alarm_pr_auc_std, 3)}
                     for r in results],
        "alarm": {
            "trip_false_alarm_budget": 0.01,
            "failing_vehicles_warned_ge_1_day": f"{int(detected.sum())}/{len(faulty)}",
            "median_lead_days": float(np.nanmedian(faulty.lead_days)) if detected.any() else None,
            "healthy_vehicles_with_false_alarm": f"{int(healthy.false_alarm.sum())}/{len(healthy)}",
            "failing_vehicles_with_alarm_before_onset": f"{int(faulty.false_alarm.sum())}/{len(faulty)}",
        },
        "confusion_matrix": {"labels": CLASSES, "counts": cm.tolist()},
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
    }
    print("[4/4]", json.dumps(summary["alarm"]))

    if not args.quick:
        img = args.out / "img"
        img.mkdir(parents=True, exist_ok=True)
        plot_degradation(df, img / "degradation.png")
        plot_ablation(results, img / "ablation.png")
        plot_confusion(cm, img / "confusion.png")
        plot_lead_time(lead, img / "lead_time.png")
        (args.out / "results.json").write_text(json.dumps(summary, indent=2))
        print(f"wrote figures and results.json to {args.out}")
    return summary


if __name__ == "__main__":
    main()
