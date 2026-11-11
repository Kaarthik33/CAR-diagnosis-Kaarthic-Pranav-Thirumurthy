// SPDX-License-Identifier: MIT
// Streaming, physics-informed feature extraction for one vehicle trip.
#pragma once

#include <array>
#include <string>
#include <vector>

#include "voltsense/rls.hpp"
#include "voltsense/rolling_stats.hpp"

namespace voltsense {

/// One telemetry sample (1/min resolution in the bundled simulator,
/// but nothing here assumes a fixed rate except the thermal model step).
struct Sample {
    double speed_kmh{};
    double current_a{};     ///< pack current, positive = discharge
    double voltage_v{};     ///< pack terminal voltage
    double soc{};           ///< state of charge, 0..1
    double cell_temp_c{};
    double motor_temp_c{};
    double coolant_temp_c{};
    double vibration_g{};   ///< drive-unit vibration RMS
};

/// Feature vector produced at the end of a trip.
struct TripFeatures {
    static constexpr std::size_t kSize = 17;
    static const std::array<const char*, kSize>& names();

    std::array<double, kSize> values{};
};

/// Consumes samples one at a time and keeps O(1) state, so the same code
/// could run on an ECU and only ship 17 numbers per trip to the cloud
/// instead of the raw time series.
class TripFeatureExtractor {
public:
    void push(const Sample& s) noexcept;
    [[nodiscard]] TripFeatures finalize() const noexcept;
    void reset() noexcept;

    [[nodiscard]] std::size_t samples() const noexcept { return speed_.count(); }

private:
    RunningStats speed_, current_, voltage_, cell_temp_, motor_temp_, vibration_;

    // Battery equivalent circuit: V = OCV(soc) - R_int * I
    //   regressors [1, I, soc]  ->  R_int = -theta[1]
    Rls<3> battery_{0.999, 1e4};

    // First-order motor thermal model (per sample step):
    //   T[k+1] - T[k] = a * (I^2 / 1e4) - b * (T[k] - T_coolant[k])
    //   regressors [I^2/1e4, T - T_cool] -> a = theta[0], b = -theta[1]
    Rls<2> thermal_{0.999, 1e4};

    // Vibration vs. speed: vib = v0 + k * speed  -> v0 is the speed-
    // independent floor, which rises with bearing wear.
    Rls<2> vib_model_{0.999, 1e4};

    bool has_prev_{false};
    Sample prev_{};
};

}  // namespace voltsense
