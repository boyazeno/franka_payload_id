// Moves to a joint configuration. Used to park the robot, to reach the trajectory
// start before a run, and to bring it somewhere convenient for mounting the tool.
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include <franka/exception.h>
#include <franka/robot.h>

#include "fpi/cli.hpp"
#include "fpi/motion_generator.hpp"
#include "fpi/safety.hpp"

namespace {

void usage() {
  std::cout <<
      "fpi_move_to --ip <fci-ip> [--q \"q1,...,q7\"] [--home] [--speed 0.15]\n"
      "\n"
      "  --home   move to a neutral, well-conditioned pose clear of the joint limits\n"
      "           (this is also a sensible place to attach or remove the tool)\n";
}

fpi::Vector7 parseJoints(const std::string& text) {
  fpi::Vector7 q{};
  std::stringstream ss(text);
  std::string cell;
  int i = 0;
  while (std::getline(ss, cell, ',')) {
    if (i >= 7) throw std::runtime_error("--q takes exactly 7 comma-separated values");
    q[i++] = std::stod(cell);
  }
  if (i != 7) throw std::runtime_error("--q takes exactly 7 comma-separated values");
  return q;
}

}  // namespace

int main(int argc, char** argv) {
  fpi::Args args(argc, argv);
  if (args.flag("help") || argc == 1) {
    usage();
    return argc == 1 ? 1 : 0;
  }

  try {
    fpi::Vector7 goal{};
    if (args.flag("home")) {
      goal = {{0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.785}};
    } else {
      goal = parseJoints(args.require("q"));
    }

    for (int j = 0; j < 7; ++j) {
      if (goal[j] < fpi::kQMin[j] || goal[j] > fpi::kQMax[j]) {
        std::cerr << "error: joint " << (j + 1) << " target " << goal[j]
                  << " is outside [" << fpi::kQMin[j] << ", " << fpi::kQMax[j] << "]\n";
        return 2;
      }
    }

    franka::Robot robot(args.require("ip"), franka::RealtimeConfig::kIgnore, 5000);
    fpi::setCollectionBehavior(robot);

    std::cout << "moving ...\n";
    MotionGenerator motion(args.number("speed", 0.15), goal);
    robot.control(motion);
    std::cout << "done.\n";
    return 0;
  } catch (const franka::Exception& e) {
    std::cerr << "franka error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
