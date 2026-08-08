// Trajectory and pose-list files produced by the Python side.
//
// Deliberately dumb formats: the excitation trajectory is already sampled to 1 kHz by
// `fpi traj export`, so this side only replays rows. Keeping the Fourier evaluation in
// Python means the robot binary has no maths worth getting wrong, and the exact
// samples that ran are the ones that were safety-checked.
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace fpi {

using Vector7 = std::array<double, 7>;

struct JointTrajectory {
  // Sampled joint positions, one row per control tick.
  std::vector<Vector7> q;
  double sample_rate_hz = 1000.0;
  int samples_per_period = 0;
  int n_periods = 0;
  // Verbatim JSON describing how it was generated; copied into the run sidecar.
  std::string source_json = "{}";

  std::size_t size() const { return q.size(); }
  double duration() const { return static_cast<double>(q.size()) / sample_rate_hz; }
};

// Reads a CSV written by `fpi traj export`:
//   line 1  : "# <json describing the trajectory>"
//   line 2  : "t,q0,q1,q2,q3,q4,q5,q6"
//   line 3+ : numeric rows
JointTrajectory loadJointTrajectory(const std::string& path);

struct StaticPose {
  Vector7 approach_from{};
  Vector7 measure_at{};
  int direction = 1;  // +1 / -1: which side the pose is approached from
};

// Reads a CSV written by `fpi poses export`:
//   line 1  : "# <json>"
//   line 2  : "direction,a0..a6,m0..m6"
std::vector<StaticPose> loadStaticPoses(const std::string& path);

}  // namespace fpi
