#include "fpi/state_log.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fpi {
namespace {

constexpr int kNJoints = 7;

void appendJointNames(std::vector<std::string>& out, const char* prefix) {
  for (int i = 0; i < kNJoints; ++i) {
    out.push_back(std::string(prefix) + "_" + std::to_string(i));
  }
}

std::string jsonArray(const double* data, std::size_t n) {
  std::ostringstream os;
  os << std::setprecision(17);
  os << "[";
  for (std::size_t i = 0; i < n; ++i) {
    if (i) os << ", ";
    os << data[i];
  }
  os << "]";
  return os.str();
}

std::string jsonEscape(const std::string& in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

}  // namespace

std::vector<std::string> recordSchema() {
  std::vector<std::string> s;
  s.reserve(kRecordSize);
  s.push_back("seq");
  s.push_back("time_s");
  s.push_back("dt_s");
  appendJointNames(s, "q");
  appendJointNames(s, "dq");
  appendJointNames(s, "q_d");
  appendJointNames(s, "dq_d");
  appendJointNames(s, "ddq_d");
  appendJointNames(s, "tau_J");
  appendJointNames(s, "tau_J_d");
  appendJointNames(s, "dtau_J");
  appendJointNames(s, "tau_ext");
  for (int i = 0; i < 16; ++i) s.push_back("O_T_EE_" + std::to_string(i));
  s.push_back("control_command_success_rate");
  s.push_back("robot_mode");
  s.push_back("errors");
  if (s.size() != kRecordSize) {
    throw std::logic_error("recordSchema() size does not match kRecordSize");
  }
  return s;
}

StateLog::StateLog(std::size_t expected_samples)
    : capacity_samples_(expected_samples) {
  // Reserve AND size the buffer so the pages are faulted in before the control loop
  // starts; a page fault inside a 1 kHz callback is a dropped packet.
  values_.assign(expected_samples * kRecordSize, 0.0);
}

void StateLog::push(const franka::RobotState& state, double period_s) noexcept {
  if (count_ >= capacity_samples_) {
    overflowed_ = true;
    return;
  }
  double* p = values_.data() + count_ * kRecordSize;
  std::size_t k = 0;

  p[k++] = static_cast<double>(count_);
  p[k++] = state.time.toSec();
  p[k++] = period_s;

  auto copy7 = [&p, &k](const std::array<double, 7>& a) {
    for (int i = 0; i < kNJoints; ++i) p[k++] = a[i];
  };
  copy7(state.q);
  copy7(state.dq);
  copy7(state.q_d);
  copy7(state.dq_d);
  copy7(state.ddq_d);
  copy7(state.tau_J);
  copy7(state.tau_J_d);
  copy7(state.dtau_J);
  copy7(state.tau_ext_hat_filtered);
  for (int i = 0; i < 16; ++i) p[k++] = state.O_T_EE[i];

  p[k++] = state.control_command_success_rate;
  p[k++] = static_cast<double>(state.robot_mode);
  // franka::Errors converts to bool (any error) and to string; pack the boolean into
  // a numeric flag so the Python side has something to gate on.
  p[k++] = static_cast<bool>(state.current_errors) ? 1.0 : 0.0;

  ++count_;
}

void captureLoadConfiguration(const franka::RobotState& state, RunMetadata& meta) {
  meta.m_ee = state.m_ee;
  meta.F_x_Cee = state.F_x_Cee;
  meta.I_ee = state.I_ee;
  meta.m_load = state.m_load;
  meta.F_x_Cload = state.F_x_Cload;
  meta.I_load = state.I_load;
  meta.F_T_NE = state.F_T_NE;
  meta.NE_T_EE = state.NE_T_EE;
}

std::string isoTimestampNow() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return std::string(buf);
}

void StateLog::write(const std::string& stem, RunMetadata meta) const {
  const std::string bin_path = stem + ".bin";
  const std::string meta_path = stem + ".meta.json";

  std::ofstream bin(bin_path, std::ios::binary | std::ios::trunc);
  if (!bin) throw std::runtime_error("cannot open " + bin_path + " for writing");
  bin.write(reinterpret_cast<const char*>(values_.data()),
            static_cast<std::streamsize>(count_ * kRecordSize * sizeof(double)));
  if (!bin) throw std::runtime_error("failed while writing " + bin_path);
  bin.close();

  const auto schema = recordSchema();
  std::ofstream js(meta_path, std::ios::trunc);
  if (!js) throw std::runtime_error("cannot open " + meta_path + " for writing");
  js << std::setprecision(17);
  js << "{\n";
  js << "  \"run_id\": \"" << jsonEscape(meta.run_id) << "\",\n";
  js << "  \"kind\": \"" << jsonEscape(meta.kind) << "\",\n";
  js << "  \"loaded\": " << (meta.loaded ? "true" : "false") << ",\n";
  js << "  \"robot_ip\": \"" << jsonEscape(meta.robot_ip) << "\",\n";
  js << "  \"libfranka_version\": \"" << jsonEscape(meta.libfranka_version) << "\",\n";
  js << "  \"robot_serial\": \"" << jsonEscape(meta.robot_serial) << "\",\n";
  js << "  \"system_version\": \"" << jsonEscape(meta.system_version) << "\",\n";
  js << "  \"collector_git_sha\": \"" << jsonEscape(meta.collector_git_sha) << "\",\n";
  js << "  \"started_at\": \"" << jsonEscape(meta.started_at) << "\",\n";
  js << "  \"finished_at\": \"" << jsonEscape(meta.finished_at) << "\",\n";
  js << "  \"sample_rate_hz\": " << meta.sample_rate_hz << ",\n";
  js << "  \"samples_per_period\": " << meta.samples_per_period << ",\n";
  js << "  \"n_periods\": " << meta.n_periods << ",\n";
  js << "  \"m_ee\": " << meta.m_ee << ",\n";
  js << "  \"F_x_Cee\": " << jsonArray(meta.F_x_Cee.data(), 3) << ",\n";
  js << "  \"I_ee\": " << jsonArray(meta.I_ee.data(), 9) << ",\n";
  js << "  \"m_load\": " << meta.m_load << ",\n";
  js << "  \"F_x_Cload\": " << jsonArray(meta.F_x_Cload.data(), 3) << ",\n";
  js << "  \"I_load\": " << jsonArray(meta.I_load.data(), 9) << ",\n";
  js << "  \"F_T_NE\": " << jsonArray(meta.F_T_NE.data(), 16) << ",\n";
  js << "  \"NE_T_EE\": " << jsonArray(meta.NE_T_EE.data(), 16) << ",\n";
  js << "  \"trajectory\": " << (meta.trajectory_json.empty() ? "{}" : meta.trajectory_json)
     << ",\n";
  js << "  \"notes\": \"" << jsonEscape(meta.notes) << "\",\n";
  js << "  \"schema\": [";
  for (std::size_t i = 0; i < schema.size(); ++i) {
    if (i) js << ", ";
    js << "\"" << schema[i] << "\"";
  }
  js << "]\n";
  js << "}\n";
  if (!js) throw std::runtime_error("failed while writing " + meta_path);
}

}  // namespace fpi
