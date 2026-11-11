// SPDX-License-Identifier: MIT
// Numerically stable single-pass statistics (Welford, 1962).
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

namespace voltsense {

/// Streaming mean / variance / min / max in O(1) memory.
/// Welford's update avoids the catastrophic cancellation of the naive
/// sum-of-squares formula, which matters on long ECU logs.
class RunningStats {
public:
    void push(double x) noexcept {
        ++n_;
        const double delta = x - mean_;
        mean_ += delta / static_cast<double>(n_);
        m2_ += delta * (x - mean_);
        if (x < min_) min_ = x;
        if (x > max_) max_ = x;
    }

    void reset() noexcept { *this = RunningStats{}; }

    [[nodiscard]] std::size_t count() const noexcept { return n_; }
    [[nodiscard]] double mean() const noexcept { return n_ ? mean_ : nan(); }
    /// Sample variance (n-1 denominator).
    [[nodiscard]] double variance() const noexcept {
        return n_ > 1 ? m2_ / static_cast<double>(n_ - 1) : nan();
    }
    [[nodiscard]] double stddev() const noexcept { return std::sqrt(variance()); }
    [[nodiscard]] double min() const noexcept { return n_ ? min_ : nan(); }
    [[nodiscard]] double max() const noexcept { return n_ ? max_ : nan(); }

private:
    static double nan() noexcept { return std::numeric_limits<double>::quiet_NaN(); }

    std::size_t n_{0};
    double mean_{0.0};
    double m2_{0.0};
    double min_{std::numeric_limits<double>::infinity()};
    double max_{-std::numeric_limits<double>::infinity()};
};

}  // namespace voltsense
