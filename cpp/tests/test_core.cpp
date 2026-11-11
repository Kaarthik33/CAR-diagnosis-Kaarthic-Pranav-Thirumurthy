// SPDX-License-Identifier: MIT
// Dependency-free unit tests (run through CTest).

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "voltsense/rls.hpp"
#include "voltsense/rolling_stats.hpp"
#include "voltsense/trip_features.hpp"

static int g_failures = 0;

#define EXPECT_NEAR(a, b, tol)                                                  \
    do {                                                                        \
        const double _a = (a), _b = (b);                                        \
        if (!(std::fabs(_a - _b) <= (tol))) {                                   \
            std::fprintf(stderr, "%s:%d: EXPECT_NEAR(%s, %s) failed: %g vs %g\n", \
                         __FILE__, __LINE__, #a, #b, _a, _b);                   \
            ++g_failures;                                                       \
        }                                                                       \
    } while (0)

#define EXPECT_TRUE(c)                                                         \
    do {                                                                       \
        if (!(c)) {                                                            \
            std::fprintf(stderr, "%s:%d: EXPECT_TRUE(%s) failed\n", __FILE__,  \
                         __LINE__, #c);                                        \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

static void test_running_stats_matches_two_pass() {
    std::mt19937 rng(42);
    std::normal_distribution<double> d(1e6, 3.0);  // large offset stresses stability
    std::vector<double> xs(10000);
    voltsense::RunningStats rs;
    for (auto& x : xs) { x = d(rng); rs.push(x); }

    double mean = 0;
    for (double x : xs) mean += x;
    mean /= static_cast<double>(xs.size());
    double var = 0;
    for (double x : xs) var += (x - mean) * (x - mean);
    var /= static_cast<double>(xs.size() - 1);

    EXPECT_NEAR(rs.mean(), mean, 1e-6);
    EXPECT_NEAR(rs.variance(), var, 1e-6);
    EXPECT_TRUE(rs.count() == xs.size());
}

static void test_running_stats_empty_is_nan() {
    voltsense::RunningStats rs;
    EXPECT_TRUE(std::isnan(rs.mean()));
    rs.push(3.0);
    EXPECT_TRUE(std::isnan(rs.variance()));
    EXPECT_NEAR(rs.min(), 3.0, 0);
    EXPECT_NEAR(rs.max(), 3.0, 0);
}

static void test_rls_recovers_linear_model() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-50, 150);
    std::normal_distribution<double> noise(0, 0.05);
    voltsense::Rls<3> rls(1.0, 1e6);
    // V = 380 - 0.08 I + 40 soc
    for (int k = 0; k < 2000; ++k) {
        const double I = u(rng), soc = 0.2 + 0.6 * (k / 2000.0);
        rls.update({1.0, I, soc}, 380 - 0.08 * I + 40 * soc + noise(rng));
    }
    EXPECT_NEAR(rls.theta()[1], -0.08, 1e-3);
    EXPECT_NEAR(rls.theta()[2], 40.0, 0.5);
}

static void test_rls_tracks_drift_with_forgetting() {
    voltsense::Rls<1> rls(0.95, 1e4);
    for (int k = 0; k < 500; ++k) rls.update({1.0}, 1.0);
    for (int k = 0; k < 200; ++k) rls.update({1.0}, 2.0);
    EXPECT_NEAR(rls.theta()[0], 2.0, 1e-3);
}

static void test_trip_extractor_identifies_physics() {
    // Synthesize a trip with known R_int and thermal parameters.
    const double R = 0.09, a = 4.0, b = 0.10;
    std::mt19937 rng(1);
    std::normal_distribution<double> n(0, 1);
    voltsense::TripFeatureExtractor fx;
    double T = 30, soc = 0.8;
    for (int k = 0; k < 400; ++k) {
        voltsense::Sample s;
        s.speed_kmh = 60 + 40 * std::sin(k / 15.0);
        s.current_a = 1.3 * s.speed_kmh + 10 * n(rng);
        soc -= s.current_a / 1e5;
        s.soc = soc;
        s.voltage_v = 340 + 60 * soc - R * s.current_a;
        s.coolant_temp_c = 25;
        s.motor_temp_c = T;
        s.cell_temp_c = 25;
        s.vibration_g = 0.5 + 0.004 * s.speed_kmh;
        fx.push(s);
        T += a * s.current_a * s.current_a / 1e4 - b * (T - s.coolant_temp_c);
    }
    const auto f = fx.finalize().values;
    EXPECT_NEAR(f[13], R, 1e-4);       // r_int_ohm
    EXPECT_NEAR(f[14], a, 1e-3);       // thermal_gain
    EXPECT_NEAR(f[15], b, 1e-4);       // thermal_conductance
    EXPECT_NEAR(f[16], 0.5, 1e-4);     // vib_floor
    EXPECT_NEAR(f[0], 400, 0);
}

static void test_short_trip_gives_nan_params() {
    voltsense::TripFeatureExtractor fx;
    for (int k = 0; k < 5; ++k) fx.push(voltsense::Sample{});
    const auto f = fx.finalize().values;
    EXPECT_TRUE(std::isnan(f[13]));
    EXPECT_TRUE(std::isnan(f[15]));
}

int main() {
    test_running_stats_matches_two_pass();
    test_running_stats_empty_is_nan();
    test_rls_recovers_linear_model();
    test_rls_tracks_drift_with_forgetting();
    test_trip_extractor_identifies_physics();
    test_short_trip_gives_nan_params();
    if (g_failures == 0) std::puts("all C++ tests passed");
    return g_failures == 0 ? 0 : 1;
}
