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
#include <string>

#include <franka/exception.h>
#include <franka/robot.h>

#include "fpi/cli.hpp"
#include "fpi/motion_generator.hpp"
#include "fpi/safety.hpp"
#include "fpi/state_log.hpp"
#include "fpi/trajectory.hpp"

namespace {

void usage() {
  std::cout <<
      "fpi_static_poses --ip <fci-ip> --poses <file.csv> --out <stem> [options]\n"
      "\n"
      "  --loaded / --bare   record which configuration this run is (required)\n"
      "  --dwell <s>         hold time per pose (default 2.0)\n"
      "  --speed <0..1>      motion speed factor (default 0.15)\n"
      "  --no-zero-load      do NOT zero the configured load (not recommended)\n"
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
    if (!args.flag("no-zero-load")) fpi::zeroLoad(robot);

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
