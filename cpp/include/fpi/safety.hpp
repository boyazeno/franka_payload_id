// Safety configuration and pre-flight checks for the collector binaries.
#pragma once

#include <array>
#include <string>
#include <vector>

#include <franka/robot.h>

#include "fpi/trajectory.hpp"

namespace fpi {

// Official Panda (FER) limits, mirroring config/panda_limits.yaml.
// Duplicated here on purpose: the robot binary must be able to refuse an unsafe file
// without depending on the Python side being present or in sync.
extern const Vector7 kQMin;
extern const Vector7 kQMax;
extern const Vector7 kQdMax;
extern const Vector7 kQddMax;
extern const Vector7 kQdddMax;
extern const Vector7 kTauMax;

// Conservative collision thresholds. Deliberately high: during collection the load is
// configured as zero while a real tool is attached, so the robot's own estimate of
// external torque is large by construction and default thresholds trip immediately.
void setCollectionBehavior(franka::Robot& robot);

// Configures the payload the robot is physically carrying: the tool's approximate mass
// and centre of mass for a loaded run, zero for a bare run.
//
// Declare it TRUTHFULLY in each run rather than zeroing both. tau_J is a physical
// link-side measurement and does not care what the controller believes, so the only
// path by which the configuration can affect the difference is through the achieved
// motion. Correct gravity compensation makes each run track its reference closely, so
// the two runs track *each other* closely -- which is what the difference method
// actually needs. Zeroing both does not make the two runs behave identically; the
// plants differ, so it makes them differently wrong, and it leaves the loaded run with
// a steady-state position error and an unmodelled payload the safety monitor can trip
// over.
void applyLoad(franka::Robot& robot, double mass, const std::array<double, 3>& com,
               const std::array<double, 9>& inertia);

// Convenience: applyLoad(robot, 0, {0,0,0}, {0...}).
void zeroLoad(franka::Robot& robot);

struct TrajectoryCheck {
  bool ok = true;
  std::vector<std::string> problems;
  double max_velocity_ratio = 0.0;
  double max_acceleration_ratio = 0.0;
  double max_jerk_ratio = 0.0;
  double max_position_excess = 0.0;
  std::string summary() const;
};

// Independent re-check of a trajectory file against the hard joint limits, performed
// on the robot PC just before execution. The Python exporter already checks far more
// (including the wall half-spaces), but this guards against a stale or hand-edited
// file reaching the robot.
TrajectoryCheck checkTrajectory(const JointTrajectory& traj, double derate_velocity,
                                double derate_acceleration, double derate_jerk);

// True if the first commanded point is close enough to the measured configuration for
// the motion generator to start without a jump.
bool nearStart(const Vector7& q_measured, const Vector7& q_first, double tolerance_rad);

}  // namespace fpi
