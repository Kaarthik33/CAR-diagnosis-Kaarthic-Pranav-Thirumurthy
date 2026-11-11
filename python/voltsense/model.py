"""Predictive-maintenance model, vehicle-grouped evaluation and lead-time
analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .features import PHYSICS_FEATURES, STAT_FEATURES

CLASSES = ["healthy", "battery", "bearing", "cooling"]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    fleet = [c for c in df.columns if c.endswith(("_roll", "_rel", "_slope"))]
    return {
        "Raw trip statistics": STAT_FEATURES,
        "+ Physics (RLS) parameters": STAT_FEATURES + PHYSICS_FEATURES,
        "+ Vehicle baseline & trend": STAT_FEATURES + PHYSICS_FEATURES + fleet,
    }


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )


@dataclass
class CvResult:
    name: str
    macro_f1: float
    macro_f1_std: float
    alarm_pr_auc: float
    alarm_pr_auc_std: float
    oof_proba: np.ndarray


def cross_validate(df: pd.DataFrame, cols: list[str], name: str, n_splits: int = 5) -> CvResult:
    """Folds are grouped by vehicle - the model is always scored on cars it has
    never seen, exactly as when deployed to a new vehicle - and stratified by
    fault type so every fold contains every failure mode."""
    y = df.target.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X = df[cols].to_numpy(dtype=float)
    oof = np.zeros((len(df), len(CLASSES)))
    f1s, aps = [], []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold, (tr, te) in enumerate(cv.split(X, df.fault, df.vehicle_id)):
        m = make_model(seed=fold).fit(X[tr], y[tr])
        proba = np.zeros((len(te), len(CLASSES)))
        proba[:, m.classes_] = m.predict_proba(X[te])
        oof[te] = proba
        f1s.append(f1_score(y[te], proba.argmax(1), average="macro",
                            labels=list(range(len(CLASSES))), zero_division=0))
        aps.append(average_precision_score(y[te] != 0, 1.0 - proba[:, 0]))
    return CvResult(name, float(np.mean(f1s)), float(np.std(f1s)),
                    float(np.mean(aps)), float(np.std(aps)), oof)


def alarm_threshold(df: pd.DataFrame, risk: np.ndarray, max_false_alarm_rate: float = 0.01) -> float:
    """Lowest threshold whose trip-level false-alarm rate on fault-free
    vehicles stays below the budget (workshop capacity is finite)."""
    healthy_risk = risk[(df.fault == "none").to_numpy()]
    return float(np.quantile(healthy_risk, 1.0 - max_false_alarm_rate))


def lead_time_report(df: pd.DataFrame, risk: np.ndarray, threshold: float, confirm: int = 2) -> pd.DataFrame:
    """Per vehicle: first day an alarm is confirmed on ``confirm`` consecutive
    trips (debouncing, like a DTC confirmation counter).

    An alarm only counts as a detection if it fires after degradation has
    physically started; an alarm before onset is scored as a false alarm,
    so a lucky early guess cannot inflate the lead time.
    """
    d = df[["vehicle_id", "day", "fault", "onset_day", "failure_day"]].copy()
    d["alarm"] = risk >= threshold
    rows = []
    for veh, g in d.sort_values("day").groupby("vehicle_id"):
        run = g.alarm.astype(int).groupby((~g.alarm).cumsum()).cumsum()
        confirmed = g.day[run >= confirm]
        fault, onset, fail = g.fault.iloc[0], g.onset_day.iloc[0], g.failure_day.iloc[0]
        if fault == "none":
            false_alarm = len(confirmed) > 0
            hit = None
        else:
            false_alarm = bool((confirmed < onset).any())
            after = confirmed[confirmed >= onset]
            hit = int(after.iloc[0]) if len(after) else None
        rows.append({
            "vehicle_id": veh,
            "fault": fault,
            "first_alarm_day": hit,
            "false_alarm": false_alarm,
            "lead_days": (fail - hit) if hit is not None else np.nan,
        })
    return pd.DataFrame(rows)


def diagnosis_confusion(df: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    y = df.target.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    return confusion_matrix(y, proba.argmax(1), labels=range(len(CLASSES)))
