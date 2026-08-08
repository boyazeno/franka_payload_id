# Robot-side collector. Only configured when BUILD_ROBOT_COLLECTOR=ON, which requires
# libfranka 0.9.2 (the last release that talks to a classic Panda).
#
# The CMake package is called `Franka` with a capital F and exports the imported target
# Franka::Franka -- not `libfranka`.
find_package(Franka 0.9 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

if(Franka_VERSION VERSION_GREATER_EQUAL 0.10)
  message(WARNING
    "Found libfranka ${Franka_VERSION}. Versions >= 0.10 target the FR3 and will refuse "
    "to connect to a classic Panda (FER). Use 0.9.2 for this robot.")
endif()

add_library(fpi_common STATIC
  src/state_log.cpp
  src/trajectory.cpp
  src/safety.cpp
  src/motion_generator.cpp
)
target_include_directories(fpi_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fpi_common PUBLIC Franka::Franka Eigen3::Eigen Threads::Threads)
target_compile_options(fpi_common PRIVATE -O2 -Wall -Wextra)

foreach(tool fpi_check fpi_move_to fpi_static_poses fpi_run_trajectory)
  add_executable(${tool} src/${tool}.cpp)
  target_link_libraries(${tool} PRIVATE fpi_common)
  target_compile_options(${tool} PRIVATE -O2 -Wall -Wextra)
endforeach()

install(TARGETS fpi_check fpi_move_to fpi_static_poses fpi_run_trajectory
        RUNTIME DESTINATION bin)
