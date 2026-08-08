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

// Zeroes the configured load. Both runs of a difference pair must be collected with
// identical (normally zero) load parameters, or the internal controller tracks
// differently in each and the torque difference is not the payload alone.
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
