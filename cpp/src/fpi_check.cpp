// Connectivity and communication-quality check. Run this FIRST on the robot PC.
//
// Identification data from a link with poor communication quality is not merely
// noisier -- when packets are dropped, Control extrapolates the commanded signal, so
// the logged state no longer corresponds to the intended trajectory. Modelled on
// libfranka's own communication_test example.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

#include "fpi/cli.hpp"
#include "fpi/safety.hpp"
#include "fpi/state_log.hpp"

namespace {

void usage() {
  std::cout <<
      "fpi_check --ip <fci-ip> [--seconds 5] [--out <stem>]\n"
      "\n"
      "  Connects, prints the configured end-effector/load parameters and the flange\n"
      "  pose, then streams robot state to measure communication quality.\n"
      "\n"
      "  --ip        FCI address (e.g. 172.16.0.2)\n"
      "  --seconds   duration of the read loop (default 5)\n"
      "  --out       write the samples to <stem>.bin/.meta.json for the FK cross-check\n"
      "  --model     also download the model library and print the flange pose\n";
}

void printMatrix(const char* name, const std::array<double, 16>& m) {
  std::cout << "  " << name << " (column-major):\n";
  for (int row = 0; row < 4; ++row) {
    std::cout << "    ";
    for (int col = 0; col < 4; ++col) {
      std::cout << std::setw(12) << std::fixed << std::setprecision(6) << m[col * 4 + row];
    }
    std::cout << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  fpi::Args args(argc, argv);
  if (args.flag("help") || argc == 1) {
    usage();
    return argc == 1 ? 1 : 0;
  }

  try {
    const std::string ip = args.require("ip");
    const double seconds = args.number("seconds", 5.0);

    std::cout << "connecting to " << ip << " ...\n";
    franka::Robot robot(ip, franka::RealtimeConfig::kIgnore);
    std::cout << "connected.\n";

    const franka::RobotState initial = robot.readOnce();

    std::cout << "\nconfigured end-effector / load (all in the FLANGE frame):\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  m_ee      = " << initial.m_ee << " kg\n";
    std::cout << "  F_x_Cee   = [" << initial.F_x_Cee[0] << ", " << initial.F_x_Cee[1]
              << ", " << initial.F_x_Cee[2] << "] m\n";
    std::cout << "  m_load    = " << initial.m_load << " kg\n";
    std::cout << "  F_x_Cload = [" << initial.F_x_Cload[0] << ", " << initial.F_x_Cload[1]
              << ", " << initial.F_x_Cload[2] << "] m\n";
    std::cout << "  m_total   = " << initial.m_total << " kg\n";
    if (initial.m_total != 0.0) {
      std::cout << "  NOTE: a load is configured. Each collection run must declare what it\n"
                   "        is actually carrying: the tool on a --loaded run, zero on a\n"
                   "        --bare one.\n";
    }
    printMatrix("F_T_EE", initial.F_T_EE);
    printMatrix("O_T_EE", initial.O_T_EE);

    std::cout << "\nmeasured joint state:\n  q     = [";
    for (int i = 0; i < 7; ++i) std::cout << initial.q[i] << (i < 6 ? ", " : "]\n");
    std::cout << "  tau_J = [";
    for (int i = 0; i < 7; ++i) std::cout << initial.tau_J[i] << (i < 6 ? ", " : "] Nm\n");
    std::cout << "  (tau_J is the raw link-side torque sensor: it INCLUDES gravity.)\n";

    if (args.flag("model")) {
      franka::Model model = robot.loadModel();
      const auto flange = model.pose(franka::Frame::kFlange, initial);
      printMatrix("model.pose(kFlange)", flange);
      const auto gravity_bare = model.gravity(initial.q, 0.0, {{0.0, 0.0, 0.0}});
      std::cout << "  gravity with load = 0: [";
      for (int i = 0; i < 7; ++i)
        std::cout << gravity_bare[i] << (i < 6 ? ", " : "] Nm\n");
    }

    // Read loop. Note this needs no real-time kernel: only robot.control() does.
    const auto expected = static_cast<std::size_t>(seconds * 1000.0) + 1000;
    fpi::StateLog log(expected);
    std::cout << "\nstreaming for " << seconds << " s ...\n";

    const auto wall_start = std::chrono::steady_clock::now();
    double last_time = initial.time.toSec();
    std::vector<double> periods_ms;
    periods_ms.reserve(expected);

    robot.read([&](const franka::RobotState& state) {
      const double now = state.time.toSec();
      const double dt = now - last_time;
      last_time = now;
      periods_ms.push_back(dt * 1e3);
      log.push(state, dt);
      return std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start)
                 .count() < seconds;
    });

    if (periods_ms.size() > 1) periods_ms.erase(periods_ms.begin());
    std::sort(periods_ms.begin(), periods_ms.end());
    const auto pct = [&periods_ms](double p) {
      if (periods_ms.empty()) return 0.0;
      const auto idx = static_cast<std::size_t>(p * (periods_ms.size() - 1));
      return periods_ms[idx];
    };
    const std::size_t late =
        std::count_if(periods_ms.begin(), periods_ms.end(), [](double v) { return v > 1.5; });

    std::cout << "\ncommunication quality over " << log.size() << " samples:\n";
    std::cout << std::setprecision(3);
    std::cout << "  state period  median " << pct(0.5) << " ms, p99 " << pct(0.99)
              << " ms, max " << (periods_ms.empty() ? 0.0 : periods_ms.back()) << " ms\n";
    std::cout << "  ticks > 1.5 ms: " << late << " ("
              << (periods_ms.empty() ? 0.0 : 100.0 * late / periods_ms.size()) << " %)\n";

    if (args.has("out")) {
      fpi::RunMetadata meta;
      meta.run_id = "check";
      meta.kind = "check";
      meta.robot_ip = ip;
      meta.started_at = fpi::isoTimestampNow();
      meta.finished_at = meta.started_at;
      fpi::captureLoadConfiguration(initial, meta);
      log.write(args.get("out", "check"), meta);
      std::cout << "  wrote " << args.get("out", "check") << ".bin\n";
    }

    const bool ok = periods_ms.empty() || (100.0 * late / periods_ms.size()) < 1.0;
    std::cout << "\nresult: " << (ok ? "OK" : "POOR -- do not collect identification data "
                                              "until this is fixed")
              << "\n";
    if (!ok) {
      std::cout << "  Check: PREEMPT_RT kernel on the host, realtime group limits,\n"
                   "         --network=host if containerised, and the NIC/cable.\n";
    }
    return ok ? 0 : 2;
  } catch (const franka::Exception& e) {
    std::cerr << "franka error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
