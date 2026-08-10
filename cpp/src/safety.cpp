#include "fpi/safety.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace fpi {

const Vector7 kQMin{{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
const Vector7 kQMax{{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};
const Vector7 kQdMax{{2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100}};
const Vector7 kQddMax{{15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0}};
const Vector7 kQdddMax{{7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0}};
const Vector7 kTauMax{{87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0}};

void setCollectionBehavior(franka::Robot& robot) {
  robot.setCollisionBehavior(
      {{40.0, 40.0, 36.0, 36.0, 24.0, 20.0, 18.0}},
      {{40.0, 40.0, 36.0, 36.0, 24.0, 20.0, 18.0}},
      {{40.0, 40.0, 36.0, 36.0, 24.0, 20.0, 18.0}},
      {{40.0, 40.0, 36.0, 36.0, 24.0, 20.0, 18.0}},
      {{40.0, 40.0, 40.0, 50.0, 50.0, 50.0}},
      {{40.0, 40.0, 40.0, 50.0, 50.0, 50.0}},
      {{40.0, 40.0, 40.0, 50.0, 50.0, 50.0}},
      {{40.0, 40.0, 40.0, 50.0, 50.0, 50.0}});
  robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
  robot.setCartesianImpedance({{3000, 3000, 3000, 300, 300, 300}});
}

void applyLoad(franka::Robot& robot, double mass, const std::array<double, 3>& com,
               const std::array<double, 9>& inertia) {
  robot.setLoad(mass, com, inertia);
}

void zeroLoad(franka::Robot& robot) {
  applyLoad(robot, 0.0, {{0.0, 0.0, 0.0}},
            {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}});
}

std::string TrajectoryCheck::summary() const {
  std::ostringstream os;
  os << "trajectory check: " << (ok ? "PASS" : "FAIL") << "\n";
  os << "  max |qd| / limit   : " << max_velocity_ratio << "\n";
  os << "  max |qdd| / limit  : " << max_acceleration_ratio << "\n";
  os << "  max |qddd| / limit : " << max_jerk_ratio << "\n";
  os << "  max position excess: " << max_position_excess << " rad";
  for (const auto& p : problems) os << "\n  PROBLEM: " << p;
  return os.str();
}

TrajectoryCheck checkTrajectory(const JointTrajectory& traj, double derate_velocity,
                                double derate_acceleration, double derate_jerk) {
  TrajectoryCheck check;
  const double dt = 1.0 / traj.sample_rate_hz;
  const std::size_t n = traj.size();

  for (std::size_t k = 0; k < n; ++k) {
    for (int j = 0; j < 7; ++j) {
      const double q = traj.q[k][j];
      check.max_position_excess =
          std::max({check.max_position_excess, kQMin[j] - q, q - kQMax[j]});
    }
  }
  if (check.max_position_excess > 0.0) {
    std::ostringstream os;
    os << "joint position limits exceeded by up to " << check.max_position_excess << " rad";
    check.problems.push_back(os.str());
    check.ok = false;
  }

  // Finite differences on the sampled path, matching how Control differentiates the
  // commanded signal (backward Euler at 1 ms).
  for (std::size_t k = 1; k + 1 < n; ++k) {
    for (int j = 0; j < 7; ++j) {
      const double qd = (traj.q[k + 1][j] - traj.q[k - 1][j]) / (2.0 * dt);
      const double qdd =
          (traj.q[k + 1][j] - 2.0 * traj.q[k][j] + traj.q[k - 1][j]) / (dt * dt);
      check.max_velocity_ratio =
          std::max(check.max_velocity_ratio, std::fabs(qd) / (kQdMax[j] * derate_velocity));
      check.max_acceleration_ratio = std::max(
          check.max_acceleration_ratio, std::fabs(qdd) / (kQddMax[j] * derate_acceleration));
    }
  }
  for (std::size_t k = 2; k + 2 < n; ++k) {
    for (int j = 0; j < 7; ++j) {
      const double qddd = (traj.q[k + 2][j] - 2.0 * traj.q[k + 1][j] +
                           2.0 * traj.q[k - 1][j] - traj.q[k - 2][j]) /
                          (2.0 * dt * dt * dt);
      check.max_jerk_ratio =
          std::max(check.max_jerk_ratio, std::fabs(qddd) / (kQdddMax[j] * derate_jerk));
    }
  }

  auto flag = [&check](const char* name, double ratio) {
    if (ratio > 1.0) {
      std::ostringstream os;
      os << name << " limit exceeded (ratio " << ratio << ")";
      check.problems.push_back(os.str());
      check.ok = false;
    }
  };
  flag("velocity", check.max_velocity_ratio);
  flag("acceleration", check.max_acceleration_ratio);
  flag("jerk", check.max_jerk_ratio);

  // The FCI requires the commanded trajectory to start and end at rest.
  if (n > 2) {
    for (int j = 0; j < 7; ++j) {
      const double v0 = std::fabs(traj.q[1][j] - traj.q[0][j]) / dt;
      const double vN = std::fabs(traj.q[n - 1][j] - traj.q[n - 2][j]) / dt;
      if (v0 > 0.05 || vN > 0.05) {
        std::ostringstream os;
        os << "joint " << (j + 1) << " does not start/end at rest (|qd| = " << v0 << " / "
           << vN << " rad/s)";
        check.problems.push_back(os.str());
        check.ok = false;
      }
    }
  }
  return check;
}

bool nearStart(const Vector7& q_measured, const Vector7& q_first, double tolerance_rad) {
  for (int j = 0; j < 7; ++j) {
    if (std::fabs(q_measured[j] - q_first[j]) > tolerance_rad) return false;
  }
  return true;
}

}  // namespace fpi
