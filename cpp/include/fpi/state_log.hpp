// Pre-allocated, allocation-free recording of franka::RobotState.
//
// The real-time callback has roughly 300 us of budget, so it must not allocate, must
// not touch the filesystem and must not print. Everything here is sized up front;
// push() only copies doubles into a vector whose capacity was reserved before the
// motion started, and the file is written after the control loop has returned.
//
// The record layout MUST stay identical to SCHEMA in
// python/franka_payload_id/data/robot_log.py. The schema is written into the sidecar
// and validated on load, so a mismatch fails loudly rather than silently shifting
// every column.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <franka/robot_state.h>

namespace fpi {

// Number of float64 values per sample. Keep in sync with robot_log.RECORD_SIZE.
//   3 scalars (seq, time_s, dt_s)
// + 9 seven-wide joint blocks (q, dq, q_d, dq_d, ddq_d, tau_J, tau_J_d, dtau_J, tau_ext)
// + 16 for the column-major O_T_EE
// + 3 scalars (success rate, robot mode, errors)
// recordSchema() throws if this constant and the generated names disagree.
inline constexpr std::size_t kRecordSize = 3 + 9 * 7 + 16 + 3;  // 85

// Field names in record order, mirroring robot_log.SCHEMA.
std::vector<std::string> recordSchema();

struct RunMetadata {
  std::string run_id;
  std::string kind;              // "trajectory" | "static" | "check"
  bool loaded = true;            // was the tool attached?
  std::string robot_ip;
  std::string libfranka_version;
  std::string robot_serial;
  std::string system_version;
  std::string collector_git_sha;
  std::string started_at;
  std::string finished_at;
  double sample_rate_hz = 1000.0;
  int samples_per_period = 0;
  int n_periods = 0;
  std::string trajectory_json = "{}";
  std::string notes;

  // Configured end-effector / load state at collection time. Both runs of a pair must
  // agree (and should be zero), or the internal controller tracks differently in each
  // and the torque difference is no longer the payload alone.
  double m_ee = 0.0;
  std::array<double, 3> F_x_Cee{};
  std::array<double, 9> I_ee{};
  double m_load = 0.0;
  std::array<double, 3> F_x_Cload{};
  std::array<double, 9> I_load{};
  std::array<double, 16> F_T_NE{};
  std::array<double, 16> NE_T_EE{};
};

class StateLog {
 public:
  // Reserves capacity for `expected_samples`; push() never allocates as long as the
  // run does not exceed it.
  explicit StateLog(std::size_t expected_samples);

  // Real-time safe: copies the fields we need into the pre-allocated buffer.
  void push(const franka::RobotState& state, double period_s) noexcept;

  std::size_t size() const noexcept { return count_; }
  bool overflowed() const noexcept { return overflowed_; }

  // Writes <stem>.bin and <stem>.meta.json. Call AFTER the control loop returns.
  void write(const std::string& stem, RunMetadata meta) const;

 private:
  std::vector<double> values_;
  std::size_t capacity_samples_;
  std::size_t count_ = 0;
  bool overflowed_ = false;
};

// Fills the load/frame fields of `meta` from a robot state.
void captureLoadConfiguration(const franka::RobotState& state, RunMetadata& meta);

std::string isoTimestampNow();

}  // namespace fpi
