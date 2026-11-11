// SPDX-License-Identifier: MIT
#include "voltsense/trip_features.hpp"

#include <limits>

namespace voltsense {

const std::array<const char*, TripFeatures::kSize>& TripFeatures::names() {
    static const std::array<const char*, kSize> kNames = {
        "n_samples",      "speed_mean",     "speed_max",       "current_mean",
        "current_max",    "voltage_std",    "cell_temp_mean",  "cell_temp_max",
        "motor_temp_mean", "motor_temp_max", "vib_mean",        "vib_std",
        "vib_max",        "r_int_ohm",      "thermal_gain",    "thermal_conductance",
        "vib_floor",
    };
    return kNames;
}

void TripFeatureExtractor::push(const Sample& s) noexcept {
    speed_.push(s.speed_kmh);
    current_.push(s.current_a);
    voltage_.push(s.voltage_v);
    cell_temp_.push(s.cell_temp_c);
    motor_temp_.push(s.motor_temp_c);
    vibration_.push(s.vibration_g);

    battery_.update({1.0, s.current_a, s.soc}, s.voltage_v);
    vib_model_.update({1.0, s.speed_kmh}, s.vibration_g);

    if (has_prev_) {
        const double i2 = prev_.current_a * prev_.current_a / 1e4;
        const double dT = s.motor_temp_c - prev_.motor_temp_c;
        thermal_.update({i2, prev_.motor_temp_c - prev_.coolant_temp_c}, dT);
    }
    prev_ = s;
    has_prev_ = true;
}

TripFeatures TripFeatureExtractor::finalize() const noexcept {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    // Require a minimum amount of excitation before trusting RLS output.
    constexpr std::size_t kMinSamples = 10;
    const bool ok = samples() >= kMinSamples;

    TripFeatures f;
    f.values = {
        static_cast<double>(samples()),
        speed_.mean(),
        speed_.max(),
        current_.mean(),
        current_.max(),
        voltage_.stddev(),
        cell_temp_.mean(),
        cell_temp_.max(),
        motor_temp_.mean(),
        motor_temp_.max(),
        vibration_.mean(),
        vibration_.stddev(),
        vibration_.max(),
        ok ? -battery_.theta()[1] : kNaN,
        ok ? thermal_.theta()[0] : kNaN,
        ok ? -thermal_.theta()[1] : kNaN,
        ok ? vib_model_.theta()[0] : kNaN,
    };
    return f;
}

void TripFeatureExtractor::reset() noexcept { *this = TripFeatureExtractor{}; }

}  // namespace voltsense
