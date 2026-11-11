// SPDX-License-Identifier: MIT
// Recursive Least Squares with exponential forgetting.
#pragma once

#include <array>
#include <cstddef>

namespace voltsense {

/// Online estimator for y = theta^T * phi, with N parameters.
///
/// This is the workhorse of on-board parameter identification in a BMS or
/// motor controller: it tracks slowly drifting physical parameters
/// (internal resistance, thermal conductance, ...) with O(N^2) work per
/// sample and no stored history, which makes it suitable for an ECU.
///
///   K      = P phi / (lambda + phi^T P phi)
///   theta += K (y - phi^T theta)
///   P      = (P - K phi^T P) / lambda
template <std::size_t N>
class Rls {
public:
    using Vec = std::array<double, N>;
    using Mat = std::array<std::array<double, N>, N>;

    explicit Rls(double lambda = 0.995, double p0 = 1e4) noexcept
        : lambda_(lambda), p0_(p0) {
        reset();
    }

    void reset() noexcept {
        theta_.fill(0.0);
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j) P_[i][j] = (i == j) ? p0_ : 0.0;
        updates_ = 0;
    }

    /// Feed one observation; returns the a-priori prediction error.
    double update(const Vec& phi, double y) noexcept {
        Vec Pphi{};
        for (std::size_t i = 0; i < N; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < N; ++j) s += P_[i][j] * phi[j];
            Pphi[i] = s;
        }
        double denom = lambda_;
        for (std::size_t i = 0; i < N; ++i) denom += phi[i] * Pphi[i];

        Vec K{};
        for (std::size_t i = 0; i < N; ++i) K[i] = Pphi[i] / denom;

        double err = y;
        for (std::size_t i = 0; i < N; ++i) err -= phi[i] * theta_[i];
        for (std::size_t i = 0; i < N; ++i) theta_[i] += K[i] * err;

        // P is symmetric, so phi^T P == (P phi)^T.
        for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                P_[i][j] = (P_[i][j] - K[i] * Pphi[j]) / lambda_;

        ++updates_;
        return err;
    }

    [[nodiscard]] const Vec& theta() const noexcept { return theta_; }
    [[nodiscard]] std::size_t updates() const noexcept { return updates_; }

private:
    double lambda_;
    double p0_;
    Vec theta_{};
    Mat P_{};
    std::size_t updates_{0};
};

}  // namespace voltsense
