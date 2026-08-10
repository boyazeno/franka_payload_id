// Stage A collection: move to each pose, dwell, average the measured torque.
//
// Each pose is visited from two opposite directions (the pose file carries the
// approach point and a direction flag). Averaging the two puts the Coulomb friction
// torque on opposite sides of its hysteresis loop, so the mean recovers the gravity
// torque alone -- the dominant error source in static identification.
//
// The dwell samples are recorded individually rather than averaged on the robot, so
// the offline side can inspect the settling behaviour and choose its own window.
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <franka/exception.h>
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

void usage() {
  std::cout <<
      "fpi_static_poses --ip <fci-ip> --poses <file.csv> --out <stem> [options]\n"
      "\n"
      "  --loaded / --bare   record which configuration this run is (required)\n"
      "  --dwell <s>         hold time per pose (default 2.0)\n"
      "  --speed <0..1>      motion speed factor (default 0.15)\n"
      "  --load-mass <kg>    tool mass to DECLARE for a --loaded run (see below)\n"
      "  --load-com x,y,z    tool centre of mass in the flange frame [m]\n"
      "  --no-zero-load      leave the configured load alone (use Desk's setting)\n"
      "\n"
      "  The pose CSV carries, per row: direction, 7 approach angles, 7 measure angles.\n"
      "  Rows are executed in order, so the two approach directions of a pose are\n"
      "  adjacent and the offline side can pair them.\n";
}

}  // namespace

int main(int argc, char** argv) {
  fpi::Args args(argc, argv);
  if (args.flag("help") || argc == 1) {
    usage();
    return argc == 1 ? 1 : 0;
  }

  try {
    const auto poses = fpi::loadStaticPoses(args.require("poses"));
    const double dwell = args.number("dwell", 2.0);
    const double speed = args.number("speed", 0.15);

    if (args.flag("loaded") == args.flag("bare")) {
      std::cerr << "error: pass exactly one of --loaded / --bare\n";
      return 1;
    }

    // Refuse pose files that stray outside the hard joint limits.
    for (std::size_t i = 0; i < poses.size(); ++i) {
      for (int j = 0; j < 7; ++j) {
        for (const auto* q : {&poses[i].approach_from, &poses[i].measure_at}) {
          if ((*q)[j] < fpi::kQMin[j] || (*q)[j] > fpi::kQMax[j]) {
            std::cerr << "error: pose " << i << " joint " << (j + 1)
                      << " is outside the joint limits\n";
            return 2;
          }
        }
      }
    }

    const std::string ip = args.require("ip");
    franka::Robot robot(ip, franka::RealtimeConfig::kEnforce, 5000);
    fpi::setCollectionBehavior(robot);
    configureLoad(robot, args, args.flag("loaded"));

    const auto dwell_samples = static_cast<std::size_t>(dwell * 1000.0) + 200;
    fpi::StateLog log(poses.size() * dwell_samples + 2000);
    const std::string started_at = fpi::isoTimestampNow();

    for (std::size_t i = 0; i < poses.size(); ++i) {
      const auto& pose = poses[i];
      std::cout << "pose " << (i + 1) << "/" << poses.size()
                << " (direction " << pose.direction << ") ... " << std::flush;

      // Back off first, then approach from the recorded side: the hysteresis state
      // depends on the direction of the LAST motion into the pose.
      MotionGenerator to_approach(speed, pose.approach_from);
      robot.control(to_approach);
      MotionGenerator to_pose(speed, pose.measure_at);
      robot.control(to_pose);

      std::size_t collected = 0;
      robot.read([&](const franka::RobotState& state) {
        log.push(state, 1e-3);
        return ++collected < dwell_samples - 200;
      });
      std::cout << collected << " samples\n";
    }

    const franka::RobotState final_state = robot.readOnce();
    fpi::RunMetadata meta;
    meta.run_id = args.get("run-id", args.require("out"));
    meta.kind = "static";
    meta.loaded = args.flag("loaded");
    meta.robot_ip = ip;
    meta.sample_rate_hz = 1000.0;
    meta.samples_per_period = static_cast<int>(dwell_samples - 200);
    meta.n_periods = static_cast<int>(poses.size());
    meta.started_at = started_at;
    meta.finished_at = fpi::isoTimestampNow();
    meta.collector_git_sha = args.get("git-sha", "");
    meta.notes = "static pose sweep; samples_per_period = dwell samples per pose";
    fpi::captureLoadConfiguration(final_state, meta);

    log.write(args.require("out"), meta);
    std::cout << "wrote " << args.require("out") << ".bin (" << log.size() << " samples)\n";
    return 0;
  } catch (const franka::Exception& e) {
    std::cerr << "franka error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
