"""Physically plausible EV fleet telemetry simulator with fault injection.

Each vehicle drives one trip per day, logged at 1 sample/minute. Signals
follow simple but real first-principles models:

* Battery:   V = OCV(SoC) - R_int * I           (Rint equivalent circuit)
* Motor:     dT = a * I^2 - b * (T - T_coolant)  (first-order thermal RC)
* Drive unit vibration grows with speed; bearing wear raises the floor
  and adds impulsive spikes.

Three degradation modes are injected, each ramping up (quadratically, so
early stages are subtle) until a failure day:

* ``cooling``  - coolant pump wear: thermal conductance ``b`` drops
* ``bearing``  - drive-unit bearing wear: vibration floor + spikes
* ``battery``  - cell degradation: internal resistance ``R_int`` rises

Vehicles differ from each other (manufacturing spread, climate, driver
style), so a fixed threshold on raw signals does not work - which is the
point of the ML pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FAULT_TYPES = ("cooling", "bearing", "battery")


@dataclass(frozen=True)
class FleetConfig:
    n_vehicles: int = 60
    n_days: int = 60
    fault_fraction: float = 0.5
    min_trip_minutes: int = 35
    max_trip_minutes: int = 80
    seed: int = 2026


@dataclass
class VehicleSpec:
    vehicle_id: str
    r_int: float          # ohm, healthy
    thermal_gain: float   # a
    conductance: float    # b, healthy
    vib_floor: float      # g, healthy
    climate_offset: float  # degC vs. fleet average
    aggressiveness: float  # driver style multiplier on speed
    fault: str | None
    onset_day: int | None
    failure_day: int | None


def _make_vehicles(cfg: FleetConfig, rng: np.random.Generator) -> list[VehicleSpec]:
    vehicles = []
    n_faulty = round(cfg.n_vehicles * cfg.fault_fraction)
    faulty_idx = set(rng.choice(cfg.n_vehicles, size=n_faulty, replace=False).tolist())
    for i in range(cfg.n_vehicles):
        fault = onset = failure = None
        if i in faulty_idx:
            fault = FAULT_TYPES[int(rng.integers(len(FAULT_TYPES)))]
            failure = int(rng.integers(min(25, cfg.n_days - 1), cfg.n_days))
            onset = max(0, failure - int(rng.integers(12, 25)))
        vehicles.append(
            VehicleSpec(
                vehicle_id=f"VIN{i:04d}",
                r_int=rng.normal(0.080, 0.008),
                thermal_gain=rng.normal(4.0, 0.3),
                conductance=rng.normal(0.10, 0.008),
                vib_floor=rng.normal(0.50, 0.05),
                climate_offset=rng.normal(0.0, 4.0),
                aggressiveness=rng.normal(1.0, 0.12),
                fault=fault,
                onset_day=onset,
                failure_day=failure,
            )
        )
    return vehicles


def _severity(v: VehicleSpec, day: int) -> float:
    """0 when healthy, ramps quadratically to 1 on the failure day."""
    if v.fault is None or day < v.onset_day:
        return 0.0
    x = (day - v.onset_day) / max(1, v.failure_day - v.onset_day)
    return float(min(1.0, x) ** 2)


def _simulate_trip(
    v: VehicleSpec, day: int, trip_minutes: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    sev = _severity(v, day)
    ambient = 17.0 + 10.0 * np.sin(2 * np.pi * day / 365.0) + v.climate_offset + rng.normal(0, 2)

    r_int = v.r_int * (1.0 + (1.2 * sev if v.fault == "battery" else 0.0))
    b = v.conductance * (1.0 - (0.55 * sev if v.fault == "cooling" else 0.0))
    vib_floor = v.vib_floor * (1.0 + (0.9 * sev if v.fault == "bearing" else 0.0))
    spike_p = 0.08 * sev if v.fault == "bearing" else 0.0

    # Speed: mean-reverting random process (urban/highway mix).
    target = rng.choice([45.0, 70.0, 110.0], p=[0.4, 0.35, 0.25]) * v.aggressiveness
    speed = np.empty(trip_minutes)
    s = 0.0
    for k in range(trip_minutes):
        s += 0.25 * (target - s) + rng.normal(0, 8)
        s = float(np.clip(s, 0.0, 160.0))
        speed[k] = s
    accel = np.diff(speed, prepend=0.0)

    current = 1.25 * speed + 2.2 * accel + rng.normal(0, 6, trip_minutes)

    soc = np.empty(trip_minutes)
    q = rng.uniform(0.55, 0.95)
    for k in range(trip_minutes):
        q -= current[k] / 9e4
        soc[k] = q
    voltage = 330.0 + 70.0 * soc - r_int * current + rng.normal(0, 0.4, trip_minutes)

    coolant = ambient + 6.0 + rng.normal(0, 0.3, trip_minutes)
    motor = np.empty(trip_minutes)
    T = ambient + rng.normal(3, 1)
    for k in range(trip_minutes):
        motor[k] = T + rng.normal(0, 0.15)
        T += v.thermal_gain * current[k] ** 2 / 1e4 - b * (T - coolant[k])

    cell = ambient + 4.0 + 0.6 * r_int / 0.08 * (np.cumsum(current**2) / 1e5) ** 0.5
    cell = cell + rng.normal(0, 0.2, trip_minutes)

    vib = vib_floor + 0.004 * speed + rng.normal(0, 0.03, trip_minutes)
    spikes = rng.random(trip_minutes) < spike_p
    vib = vib + spikes * rng.uniform(0.3, 0.9, trip_minutes)

    return {
        "minute": np.arange(trip_minutes),
        "speed_kmh": speed,
        "current_a": current,
        "voltage_v": voltage,
        "soc": soc,
        "cell_temp_c": cell,
        "motor_temp_c": motor,
        "coolant_temp_c": coolant,
        "vibration_g": vib,
    }


def simulate_fleet(cfg: FleetConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (telemetry, vehicles) data frames.

    ``telemetry`` is sorted by vehicle and trip, exactly as an in-vehicle
    logger would write it. ``vehicles`` is the ground truth (fault type,
    onset and failure day) used only for labelling and evaluation.
    """
    cfg = cfg or FleetConfig()
    rng = np.random.default_rng(cfg.seed)
    vehicles = _make_vehicles(cfg, rng)
    frames = []
    for v in vehicles:
        last_day = v.failure_day if v.failure_day is not None else cfg.n_days - 1
        for day in range(last_day + 1):
            minutes = int(rng.integers(cfg.min_trip_minutes, cfg.max_trip_minutes + 1))
            trip = _simulate_trip(v, day, minutes, rng)
            df = pd.DataFrame(trip)
            df.insert(0, "day", day)
            df.insert(0, "trip_id", f"{v.vehicle_id}-D{day:03d}")
            df.insert(0, "vehicle_id", v.vehicle_id)
            frames.append(df)
    telemetry = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(
        [
            {
                "vehicle_id": v.vehicle_id,
                "fault": v.fault or "none",
                "onset_day": v.onset_day,
                "failure_day": v.failure_day,
            }
            for v in vehicles
        ]
    )
    return telemetry, meta
