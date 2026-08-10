// Plays a pre-computed excitation trajectory and logs robot state at 1 kHz.
//
// Real-time discipline in the control callback: no allocation, no I/O, no printing.
// Everything is reserved before robot.control() is entered and the log is written
// after it returns. The callback has roughly 300 us of budget.
#include <algorithm>
#include <atomic>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <franka/exception.h>
#include <franka/log.h>
#include <franka/robot.h>

#include "fpi/cli.hpp"
#include "fpi/motion_generator.hpp"
#include "fpi/safety.hpp"
#include "fpi/state_log.hpp"
#include "fpi/trajectory.hpp"

namespace {

// Parses "--load-com x,y,z"; returns {0,0,0} when absent.
std::array<double, 3> parseCom(const fpi::Args& args) {
  std::array<double, 3> com{{0.0, 0.0, 0.0}};
  const std::string text = args.get("load-com", "");
  if (text.empty()) return com;
  std::stringstream ss(text);
  std::string cell;
  int i = 0;
  while (std::getline(ss, cell, ',')) {
    if (i >= 3) throw std::runtime_error("--load-com takes exactly 3 comma-separated values");
    com[i++] = std::stod(cell);
  }
  if (i != 3) throw std::runtime_error("--load-com takes exactly 3 comma-separated values");
  return com;
}

// Declares the payload the robot is physically carrying for this run.
void configureLoad(franka::Robot& robot, const fpi::Args& args, bool loaded) {
  if (args.flag("no-zero-load")) {
    std::cout << "leaving the configured load untouched (--no-zero-load); make sure "
                 "Desk matches what is physically attached\n";
    return;
  }
  const double mass = loaded ? args.number("load-mass", 0.0) : 0.0;
  const auto com = loaded ? parseCom(args) : std::array<double, 3>{{0.0, 0.0, 0.0}};
  fpi::applyLoad(robot, mass, com, {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});

  if (loaded && mass <= 0.0) {
    std::cout << "WARNING: running LOADED with the configured load set to zero.\n"
                 "  Gravity compensation will not account for the tool, the arm will sag\n"
                 "  under the impedance error, and the robot may abort with\n"
                 "  tau_J_range_violation. Pass --load-mass (and --load-com) with the\n"
                 "  tool's approximate values from config/tool.yaml.\n";
  } else {
    std::cout << "configured load: " << mass << " kg at [" << com[0] << ", " << com[1]
              << ", " << com[2] << "] m (flange frame)\n";
  }
}

}  // namespace


namespace {

std::atomic<bool> g_stop{false};
void handleSignal(int) { g_stop.store(true); }

void usage() {
  std::cout <<
      "fpi_run_trajectory --ip <fci-ip> --traj <file.csv> --out <stem> [options]\n"
      "\n"
      "  --loaded / --bare   record which configuration this run is (required)\n"
      "  --speed <0..1>      speed factor for the approach move (default 0.15)\n"
      "  --blend <s>         seconds over which the residual offset between the\n"
      "                      measured start pose and the trajectory's first point is\n"
      "                      faded out (default 1.0). The FCI requires the commanded\n"
      "                      position to equal the measured one at t=0; commanding the\n"
      "                      trajectory directly makes the rate limiter lunge.\n"
      "  --dry-run <0..1>    scale the excitation amplitude about its start pose;\n"
      "                      use 0.2 for the first hardware run\n"
      "  --simulate          generate the log without touching a robot, for testing\n"
      "                      the file format off-hardware\n"
      "\n"
      "  --load-mass <kg>    tool mass to DECLARE to the robot for a --loaded run.\n"
      "  --load-com x,y,z    tool centre of mass in the flange frame [m].\n"
      "                      Declare the payload truthfully: tau_J is a physical\n"
      "                      measurement and does not care what the controller\n"
      "                      believes, but correct gravity compensation is what keeps\n"
      "                      both runs tracking the same reference -- and running with\n"
      "                      an unmodelled tool invites tau_J_range_violation.\n"
      "                      A --bare run always declares zero.\n"
      "  --no-zero-load      leave the configured load alone (use Desk's setting).\n";
}

// Writes a plausible log with no robot attached, so the record format and the Python
// parser can be exercised in the robot image itself.
int simulate(const fpi::Args& args, const fpi::JointTrajectory& traj) {
  fpi::StateLog log(traj.size() + 16);
  franka::RobotState state{};
  for (std::size_t k = 0; k < traj.size(); ++k) {
    state.q = traj.q[k];
    state.q_d = traj.q[k];
    state.time = franka::Duration(static_cast<uint64_t>(k));
    state.control_command_success_rate = 1.0;
    state.robot_mode = franka::RobotMode::kMove;
    log.push(state, 1.0 / traj.sample_rate_hz);
  }
  fpi::RunMetadata meta;
  meta.run_id = args.get("out", "simulated");
  meta.kind = "trajectory";
  meta.loaded = args.flag("loaded");
  meta.robot_ip = "simulated";
  meta.sample_rate_hz = traj.sample_rate_hz;
  meta.samples_per_period = traj.samples_per_period;
  meta.n_periods = traj.n_periods;
  meta.trajectory_json = traj.source_json;
  meta.started_at = fpi::isoTimestampNow();
  meta.finished_at = meta.started_at;
  meta.notes = "SIMULATED -- no robot involved";
  log.write(args.require("out"), meta);
  std::cout << "wrote " << args.require("out") << ".bin (" << log.size()
            << " simulated samples)\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  fpi::Args args(argc, argv);
  if (args.flag("help") || argc == 1) {
    usage();
    return argc == 1 ? 1 : 0;
  }

  try {
    const fpi::JointTrajectory traj = fpi::loadJointTrajectory(args.require("traj"));
    std::cout << "loaded " << traj.size() << " samples (" << traj.duration() << " s at "
              << traj.sample_rate_hz << " Hz)\n";

    if (!args.flag("loaded") && !args.flag("bare")) {
      std::cerr << "error: pass exactly one of --loaded / --bare so the run can be paired\n";
      return 1;
    }
    if (args.flag("loaded") && args.flag("bare")) {
      std::cerr << "error: --loaded and --bare are mutually exclusive\n";
      return 1;
    }

    const double dry = args.number("dry-run", 1.0);
    if (dry <= 0.0 || dry > 1.0) {
      std::cerr << "error: --dry-run must lie in (0, 1]\n";
      return 1;
    }

    fpi::JointTrajectory played = traj;
    if (dry < 1.0) {
      // Shrink the excursion about the first point; keeps the start/end at rest.
      for (auto& row : played.q) {
        for (int j = 0; j < 7; ++j) {
          row[j] = traj.q[0][j] + dry * (row[j] - traj.q[0][j]);
        }
      }
      std::cout << "DRY RUN: excitation scaled to " << dry << " of full amplitude\n";
    }

    const auto check = fpi::checkTrajectory(played, args.number("derate-velocity", 1.0),
                                            args.number("derate-acceleration", 1.0),
                                            args.number("derate-jerk", 1.0));
    std::cout << check.summary() << "\n";
    if (!check.ok) {
      std::cerr << "refusing to execute a trajectory that violates the joint limits\n";
      return 2;
    }

    if (args.flag("simulate")) return simulate(args, played);

    const std::string ip = args.require("ip");
    std::signal(SIGINT, handleSignal);

    // A generous log_size means a reflex still leaves several seconds of post-mortem
    // data in the ControlException.
    franka::Robot robot(ip, franka::RealtimeConfig::kEnforce, 5000);
    fpi::setCollectionBehavior(robot);
    configureLoad(robot, args, args.flag("loaded"));

    // Approach the trajectory start smoothly. The tolerance below is an "is it already
    // exactly there" test, NOT a "close enough to jump" test: the FCI requires the
    // commanded position to equal the measured one at t = 0, and any step is turned by
    // libfranka's rate limiter into a maximum-acceleration lunge toward the start.
    constexpr double kAlreadyThereRad = 0.01;
    franka::RobotState initial = robot.readOnce();
    if (!fpi::nearStart(initial.q, played.q.front(), kAlreadyThereRad)) {
      std::cout << "moving to the trajectory start ...\n";
      MotionGenerator to_start(args.number("speed", 0.15), played.q.front());
      robot.control(to_start);
      initial = robot.readOnce();
    }

    // Whatever residual remains after the approach is absorbed by a smooth blend rather
    // than commanded away in one tick. The first commanded point is then EXACTLY the
    // measured configuration, and the raised-cosine weight has zero derivative at both
    // ends, so neither the start of the blend nor its end introduces a velocity step.
    std::array<double, 7> offset{};
    double worst_offset = 0.0;
    for (int j = 0; j < 7; ++j) {
      offset[j] = initial.q[j] - played.q.front()[j];
      worst_offset = std::max(worst_offset, std::fabs(offset[j]));
    }
    if (worst_offset > 0.1) {
      std::cerr << "error: still " << worst_offset << " rad from the trajectory start "
                << "after the approach move; refusing to execute\n";
      return 2;
    }

    const double blend_s = args.number("blend", 1.0);
    const double period_s = played.samples_per_period > 0
                                ? played.samples_per_period / played.sample_rate_hz
                                : played.duration();
    if (blend_s >= period_s) {
      std::cerr << "error: --blend " << blend_s << " s must be shorter than one period ("
                << period_s << " s), so the blended samples fall entirely inside the\n"
                   "       settling period that the offline stage discards\n";
      return 2;
    }
    std::cout << "start offset " << worst_offset << " rad, blended out over " << blend_s
              << " s\n";

    fpi::StateLog log(played.size() + 2000);
    std::size_t index = 0;
    double elapsed = 0.0;

    std::cout << "executing ...\n";
    const std::string started_at = fpi::isoTimestampNow();

    try {
      robot.control([&](const franka::RobotState& state,
                        franka::Duration period) -> franka::JointPositions {
        elapsed += period.toSec();
        log.push(state, period.toSec());

        if (index >= played.size() || g_stop.load()) {
          return franka::MotionFinished(franka::JointPositions(played.q.back()));
        }

        // Raised-cosine blend: weight 1 at t=0 (command == measured start), 0 after
        // blend_s, with zero slope at both ends.
        double weight = 0.0;
        if (elapsed < blend_s) {
          weight = 0.5 * (1.0 + std::cos(M_PI * elapsed / blend_s));
        }

        std::array<double, 7> target = played.q[index];
        for (int j = 0; j < 7; ++j) target[j] += offset[j] * weight;

        franka::JointPositions command(target);
        ++index;
        return command;
      });
    } catch (const franka::ControlException& e) {
      std::cerr << "control exception: " << e.what() << "\n";
      const std::string dump = args.require("out") + ".controlexception.csv";
      std::ofstream(dump) << franka::logToCSV(e.log);
      std::cerr << "wrote post-mortem log to " << dump << "\n";
      robot.automaticErrorRecovery();
      return 3;
    }

    if (log.overflowed()) {
      std::cerr << "WARNING: the state buffer overflowed; the run is truncated\n";
    }

    const franka::RobotState final_state = robot.readOnce();
    fpi::RunMetadata meta;
    meta.run_id = args.get("run-id", args.require("out"));
    meta.kind = "trajectory";
    meta.loaded = args.flag("loaded");
    meta.robot_ip = ip;
    meta.sample_rate_hz = played.sample_rate_hz;
    meta.samples_per_period = played.samples_per_period;
    meta.n_periods = played.n_periods;
    meta.trajectory_json = played.source_json;
    meta.started_at = started_at;
    meta.finished_at = fpi::isoTimestampNow();
    meta.collector_git_sha = args.get("git-sha", "");
    meta.notes = dry < 1.0 ? "dry run, amplitude scaled" : "";
    fpi::captureLoadConfiguration(final_state, meta);

    log.write(args.require("out"), meta);
    std::cout << "wrote " << args.require("out") << ".bin (" << log.size() << " samples, "
              << elapsed << " s)\n";
    return 0;
  } catch (const franka::Exception& e) {
    std::cerr << "franka error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
