"""Trip-level feature extraction (C++ fast path + Python reference) and
causal fleet-level features.

The C++ extractor (``voltsense_extract``) is the production path. The pure
Python implementation below mirrors it line by line; the test-suite checks
that both agree, so the C++ code is verified against an independent,
readable reference.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "n_samples", "speed_mean", "speed_max", "current_mean", "current_max",
    "voltage_std", "cell_temp_mean", "cell_temp_max", "motor_temp_mean",
    "motor_temp_max", "vib_mean", "vib_std", "vib_max", "r_int_ohm",
    "thermal_gain", "thermal_conductance", "vib_floor",
]
PHYSICS_FEATURES = ["r_int_ohm", "thermal_gain", "thermal_conductance", "vib_floor"]
STAT_FEATURES = [f for f in FEATURE_NAMES if f not in PHYSICS_FEATURES and f != "n_samples"]

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Python reference implementation
# --------------------------------------------------------------------------- #
class _Rls:
    def __init__(self, n: int, lam: float = 0.999, p0: float = 1e4) -> None:
        self.lam = lam
        self.theta = np.zeros(n)
        self.P = np.eye(n) * p0

    def update(self, phi: np.ndarray, y: float) -> None:
        Pphi = self.P @ phi
        K = Pphi / (self.lam + phi @ Pphi)
        self.theta = self.theta + K * (y - phi @ self.theta)
        self.P = (self.P - np.outer(K, Pphi)) / self.lam


def extract_trip_features_py(trip: pd.DataFrame) -> list[float]:
    """Reference feature extraction for one trip (mirrors trip_features.cpp)."""
    n = len(trip)
    sp, cur, volt = trip.speed_kmh.to_numpy(), trip.current_a.to_numpy(), trip.voltage_v.to_numpy()
    soc, cell = trip.soc.to_numpy(), trip.cell_temp_c.to_numpy()
    mot, cool, vib = trip.motor_temp_c.to_numpy(), trip.coolant_temp_c.to_numpy(), trip.vibration_g.to_numpy()

    bat, th, vm = _Rls(3), _Rls(2), _Rls(2)
    for k in range(n):
        bat.update(np.array([1.0, cur[k], soc[k]]), volt[k])
        vm.update(np.array([1.0, sp[k]]), vib[k])
        if k > 0:
            th.update(np.array([cur[k - 1] ** 2 / 1e4, mot[k - 1] - cool[k - 1]]), mot[k] - mot[k - 1])

    ok = n >= 10
    std = lambda x: float(np.std(x, ddof=1)) if n > 1 else np.nan
    return [
        float(n), sp.mean(), sp.max(), cur.mean(), cur.max(), std(volt),
        cell.mean(), cell.max(), mot.mean(), mot.max(), vib.mean(), std(vib), vib.max(),
        -bat.theta[1] if ok else np.nan,
        th.theta[0] if ok else np.nan,
        -th.theta[1] if ok else np.nan,
        vm.theta[0] if ok else np.nan,
    ]


def extract_features_py(telemetry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (veh, trip), g in telemetry.groupby(["vehicle_id", "trip_id"], sort=False):
        rows.append([veh, trip, int(g.day.iloc[0]), *extract_trip_features_py(g)])
    return pd.DataFrame(rows, columns=["vehicle_id", "trip_id", "day", *FEATURE_NAMES])


# --------------------------------------------------------------------------- #
# C++ fast path
# --------------------------------------------------------------------------- #
def find_extractor() -> Path | None:
    env = os.environ.get("VOLTSENSE_EXTRACT")
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "build" / "voltsense_extract", REPO_ROOT / "build" / "Release" / "voltsense_extract.exe"]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    found = shutil.which("voltsense_extract")
    return Path(found) if found else None


def extract_features_cpp(telemetry: pd.DataFrame, binary: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "telemetry.csv", Path(tmp) / "features.csv"
        telemetry.to_csv(src, index=False, float_format="%.10g")
        subprocess.run([str(binary), str(src), str(dst)], check=True)
        return pd.read_csv(dst)


def extract_features(telemetry: pd.DataFrame, prefer_cpp: bool = True) -> pd.DataFrame:
    binary = find_extractor() if prefer_cpp else None
    if binary is not None:
        return extract_features_cpp(telemetry, binary)
    print("[voltsense] C++ extractor not found - using the (slower) Python reference.")
    return extract_features_py(telemetry)


# --------------------------------------------------------------------------- #
# Fleet-level, strictly causal features
# --------------------------------------------------------------------------- #
def add_fleet_features(feats: pd.DataFrame, baseline_trips: int = 7, window: int = 5) -> pd.DataFrame:
    """Add per-vehicle baseline-relative and trend features.

    Every feature on day *d* uses only trips from days <= *d*: the model can
    run online without peeking into the future (no target leakage).

    * ``<f>_roll``  - rolling median over the last ``window`` trips
                      (denoises single-trip parameter estimates)
    * ``<f>_rel``   - rolling value relative to the vehicle's own baseline
                      (first ``baseline_trips`` trips) -> removes the
                      vehicle-to-vehicle manufacturing spread
    * ``<f>_slope`` - change of the rolling value over the last ``window`` trips
    """
    df = feats.sort_values(["vehicle_id", "day"]).reset_index(drop=True).copy()
    g = df.groupby("vehicle_id", sort=False)
    tracked = PHYSICS_FEATURES + ["vib_max", "vib_std", "motor_temp_max", "voltage_std"]
    for f in tracked:
        roll = g[f].transform(lambda s: s.rolling(window, min_periods=1).median())
        # Expanding median, frozen once `baseline_trips` trips are seen.
        base = g[f].transform(
            lambda s: s.expanding().median().where(np.arange(len(s)) < baseline_trips).ffill()
        )
        df[f"{f}_roll"] = roll
        df[f"{f}_rel"] = roll / base - 1.0
        df[f"{f}_slope"] = roll - roll.groupby(df.vehicle_id).shift(window)
    return df


def label_trips(df: pd.DataFrame, vehicles: pd.DataFrame, horizon_days: int = 7) -> pd.DataFrame:
    """Target: which component (if any) will fail within ``horizon_days``."""
    out = df.merge(vehicles, on="vehicle_id", how="left")
    days_to_fail = out.failure_day - out.day
    imminent = out.failure_day.notna() & (days_to_fail <= horizon_days)
    out["days_to_failure"] = days_to_fail
    out["target"] = np.where(imminent, out.fault, "healthy")
    return out
