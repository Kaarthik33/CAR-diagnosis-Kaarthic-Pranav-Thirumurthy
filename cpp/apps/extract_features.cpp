// SPDX-License-Identifier: MIT
//
// voltsense_extract: stream a raw fleet telemetry CSV and emit one
// physics-informed feature row per (vehicle, trip).
//
//   voltsense_extract <telemetry.csv> <features.csv>
//   voltsense_extract - -            (stdin -> stdout)
//
// Input must be grouped by vehicle_id, trip_id (the order a logger writes).
// Memory use is constant regardless of file size.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "voltsense/trip_features.hpp"

namespace {

std::vector<std::string> split(const std::string& line, char sep = ',') {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == sep) {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

double to_double(const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str()) throw std::runtime_error("not a number: '" + s + "'");
    return v;
}

void write_number(std::ostream& os, double v) {
    if (std::isnan(v)) {
        os << "";  // empty cell == missing, pandas reads it as NaN
    } else {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.10g", v);
        os << buf;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <telemetry.csv|-> <features.csv|->\n";
        return 2;
    }
    std::ios::sync_with_stdio(false);

    std::ifstream fin;
    std::ofstream fout;
    std::istream* in = &std::cin;
    std::ostream* out = &std::cout;
    if (std::string(argv[1]) != "-") {
        fin.open(argv[1]);
        if (!fin) { std::cerr << "cannot open " << argv[1] << "\n"; return 1; }
        in = &fin;
    }
    if (std::string(argv[2]) != "-") {
        fout.open(argv[2]);
        if (!fout) { std::cerr << "cannot open " << argv[2] << "\n"; return 1; }
        out = &fout;
    }

    const auto t0 = std::chrono::steady_clock::now();

    std::string line;
    if (!std::getline(*in, line)) { std::cerr << "empty input\n"; return 1; }
    const auto header = split(line);
    std::unordered_map<std::string, std::size_t> col;
    for (std::size_t i = 0; i < header.size(); ++i) col[header[i]] = i;

    const char* required[] = {"vehicle_id", "trip_id", "day", "speed_kmh",
                              "current_a", "voltage_v", "soc", "cell_temp_c",
                              "motor_temp_c", "coolant_temp_c", "vibration_g"};
    for (const char* r : required) {
        if (!col.count(r)) { std::cerr << "missing column: " << r << "\n"; return 1; }
    }

    *out << "vehicle_id,trip_id,day";
    for (const char* n : voltsense::TripFeatures::names()) *out << ',' << n;
    *out << '\n';

    voltsense::TripFeatureExtractor fx;
    std::string cur_vehicle, cur_trip, cur_day;
    std::size_t rows = 0, trips = 0;

    auto flush = [&] {
        if (fx.samples() == 0) return;
        *out << cur_vehicle << ',' << cur_trip << ',' << cur_day;
        for (double v : fx.finalize().values) { *out << ','; write_number(*out, v); }
        *out << '\n';
        ++trips;
        fx.reset();
    };

    try {
        while (std::getline(*in, line)) {
            if (line.empty()) continue;
            const auto f = split(line);
            if (f.size() < header.size()) throw std::runtime_error("short row");
            const std::string& veh = f[col["vehicle_id"]];
            const std::string& trip = f[col["trip_id"]];
            if (veh != cur_vehicle || trip != cur_trip) {
                flush();
                cur_vehicle = veh;
                cur_trip = trip;
                cur_day = f[col["day"]];
            }
            voltsense::Sample s;
            s.speed_kmh = to_double(f[col["speed_kmh"]]);
            s.current_a = to_double(f[col["current_a"]]);
            s.voltage_v = to_double(f[col["voltage_v"]]);
            s.soc = to_double(f[col["soc"]]);
            s.cell_temp_c = to_double(f[col["cell_temp_c"]]);
            s.motor_temp_c = to_double(f[col["motor_temp_c"]]);
            s.coolant_temp_c = to_double(f[col["coolant_temp_c"]]);
            s.vibration_g = to_double(f[col["vibration_g"]]);
            fx.push(s);
            ++rows;
        }
        flush();
    } catch (const std::exception& e) {
        std::cerr << "error at data row " << rows + 1 << ": " << e.what() << "\n";
        return 1;
    }

    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cerr << "voltsense_extract: " << rows << " samples -> " << trips << " trips in "
              << secs << " s (" << static_cast<long>(static_cast<double>(rows) / (secs > 0 ? secs : 1e-9))
              << " samples/s)\n";
    return 0;
}
