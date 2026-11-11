import numpy as np
import pandas as pd
import pytest
from voltsense.features import (
    FEATURE_NAMES,
    add_fleet_features,
    extract_features_cpp,
    extract_features_py,
    find_extractor,
    label_trips,
)
from voltsense.model import CLASSES, lead_time_report
from voltsense.simulator import FleetConfig, simulate_fleet


@pytest.fixture(scope="module")
def small_fleet():
    return simulate_fleet(FleetConfig(n_vehicles=6, n_days=30, seed=1))


def test_simulator_is_deterministic():
    a, _ = simulate_fleet(FleetConfig(n_vehicles=2, n_days=3, seed=5))
    b, _ = simulate_fleet(FleetConfig(n_vehicles=2, n_days=3, seed=5))
    pd.testing.assert_frame_equal(a, b)


def test_failing_vehicles_stop_reporting_after_failure(small_fleet):
    tele, meta = small_fleet
    for _, v in meta[meta.fault != "none"].iterrows():
        assert tele[tele.vehicle_id == v.vehicle_id].day.max() == v.failure_day


def test_python_reference_recovers_injected_battery_fault():
    tele, meta = simulate_fleet(FleetConfig(n_vehicles=12, n_days=40, fault_fraction=1.0, seed=3))
    feats = extract_features_py(tele).merge(meta, on="vehicle_id")
    bat = feats[feats.fault == "battery"]
    if bat.empty:
        pytest.skip("no battery fault drawn for this seed")
    early = bat[bat.day < bat.onset_day].r_int_ohm.median()
    late = bat[bat.day >= bat.failure_day - 1].r_int_ohm.median()
    assert late > 1.6 * early  # simulator raises R_int by 2.2x at failure


@pytest.mark.skipif(find_extractor() is None, reason="C++ extractor not built")
def test_cpp_matches_python_reference(small_fleet):
    tele, _ = small_fleet
    tele = tele[tele.day < 5]
    cpp = extract_features_cpp(tele, find_extractor())
    py = extract_features_py(tele)
    assert list(cpp.trip_id) == list(py.trip_id)
    np.testing.assert_allclose(cpp[FEATURE_NAMES].to_numpy(), py[FEATURE_NAMES].to_numpy(), rtol=1e-6, atol=1e-8)


def test_fleet_features_are_causal(small_fleet):
    """Changing the future must not change today's features (no leakage)."""
    tele, _ = small_fleet
    feats = extract_features_py(tele[tele.day < 20])
    full = add_fleet_features(feats)
    tampered = feats.copy()
    tampered.loc[tampered.day >= 15, "r_int_ohm"] *= 10
    cut = add_fleet_features(tampered)
    past = full.day < 15
    cols = [c for c in full.columns if c.startswith("r_int_ohm_")]
    pd.testing.assert_frame_equal(full.loc[past, cols], cut.loc[past, cols])


def test_labels(small_fleet):
    tele, meta = small_fleet
    df = label_trips(extract_features_py(tele[tele.day < 3]), meta)
    assert set(df.target) <= set(CLASSES)


def test_lead_time_ignores_alarms_before_onset():
    df = pd.DataFrame({
        "vehicle_id": ["A"] * 10, "day": range(10), "fault": ["bearing"] * 10,
        "onset_day": [5] * 10, "failure_day": [9] * 10,
    })
    risk = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    rep = lead_time_report(df, risk, threshold=0.5)
    assert rep.first_alarm_day.iloc[0] == 7  # confirmed on 2nd consecutive trip after onset
    assert rep.lead_days.iloc[0] == 2
    assert bool(rep.false_alarm.iloc[0])
