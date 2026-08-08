#include "fpi/trajectory.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fpi {
namespace {

std::vector<double> parseRow(const std::string& line) {
  std::vector<double> out;
  std::stringstream ss(line);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    if (cell.empty()) continue;
    out.push_back(std::stod(cell));
  }
  return out;
}

// Pulls "key": <number> out of the header JSON without a JSON library. Only used for
// a handful of scalars that are also re-derived from the data, so a miss is harmless.
double jsonNumber(const std::string& json, const std::string& key, double fallback) {
  const std::string needle = "\"" + key + "\"";
  const auto pos = json.find(needle);
  if (pos == std::string::npos) return fallback;
  const auto colon = json.find(':', pos + needle.size());
  if (colon == std::string::npos) return fallback;
  try {
    return std::stod(json.substr(colon + 1));
  } catch (const std::exception&) {
    return fallback;
  }
}

std::string readHeader(std::ifstream& in) {
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("file is empty");
  if (!line.empty() && line[0] == '#') {
    return line.substr(1);
  }
  // No header comment: rewind so the caller still sees this line.
  in.seekg(0);
  return "{}";
}

}  // namespace

JointTrajectory loadJointTrajectory(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open trajectory file " + path);

  JointTrajectory traj;
  traj.source_json = readHeader(in);
  traj.sample_rate_hz = jsonNumber(traj.source_json, "sample_rate_hz", 1000.0);
  traj.samples_per_period =
      static_cast<int>(jsonNumber(traj.source_json, "samples_per_period", 0));
  traj.n_periods = static_cast<int>(jsonNumber(traj.source_json, "n_periods", 0));

  std::string line;
  std::getline(in, line);  // column header row
  std::size_t row = 2;
  while (std::getline(in, line)) {
    ++row;
    if (line.empty()) continue;
    const auto values = parseRow(line);
    if (values.size() != 8) {
      throw std::runtime_error(path + ":" + std::to_string(row) + ": expected 8 columns (t + 7 joints), got " +
                               std::to_string(values.size()));
    }
    Vector7 q{};
    for (int i = 0; i < 7; ++i) q[i] = values[i + 1];
    traj.q.push_back(q);
  }
  if (traj.q.empty()) throw std::runtime_error(path + " contains no trajectory rows");
  return traj;
}

std::vector<StaticPose> loadStaticPoses(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open pose file " + path);

  readHeader(in);
  std::string line;
  std::getline(in, line);  // column header row

  std::vector<StaticPose> poses;
  std::size_t row = 2;
  while (std::getline(in, line)) {
    ++row;
    if (line.empty()) continue;
    const auto values = parseRow(line);
    if (values.size() != 15) {
      throw std::runtime_error(path + ":" + std::to_string(row) +
                               ": expected 15 columns (direction + 7 approach + 7 measure), got " +
                               std::to_string(values.size()));
    }
    StaticPose pose;
    pose.direction = static_cast<int>(values[0]);
    for (int i = 0; i < 7; ++i) {
      pose.approach_from[i] = values[1 + i];
      pose.measure_at[i] = values[8 + i];
    }
    poses.push_back(pose);
  }
  if (poses.empty()) throw std::runtime_error(path + " contains no poses");
  return poses;
}

}  // namespace fpi
