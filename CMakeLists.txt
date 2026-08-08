cmake_minimum_required(VERSION 3.16)
project(franka_payload_id LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# The robot-side collector needs libfranka, which only exists on the robot PC (and in
# docker/Dockerfile.robot). Default OFF so this repo configures and builds on a
# development machine with no libfranka and no robot.
option(BUILD_ROBOT_COLLECTOR "Build the libfranka-based data collector" OFF)

if(BUILD_ROBOT_COLLECTOR)
  add_subdirectory(cpp)
else()
  message(STATUS
    "franka_payload_id: robot collector DISABLED. "
    "Configure with -DBUILD_ROBOT_COLLECTOR=ON on a machine that has libfranka 0.9.2.")
endif()
