cmake_minimum_required(VERSION 3.16)

project(vision_target_tracker
  VERSION 1.0.0
  DESCRIPTION "Real-time correlation-filter target tracker with Kalman prediction and PID gimbal control"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VTT_BUILD_APPS "Build the vtt and vtt_synth executables" ON)
option(VTT_BUILD_TESTS "Build unit and integration tests" ON)
option(VTT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(VTT_ENABLE_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(VTT_ENABLE_COVERAGE "Build with gcov instrumentation" OFF)
option(VTT_NATIVE_ARCH "Optimise for the host CPU (-march=native)" OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio highgui)

include(cmake/CompilerOptions.cmake)

add_library(vtt_core STATIC
  src/types.cpp
  src/spectral.cpp
  src/features.cpp
  src/mosse.cpp
  src/kcf.cpp
  src/scale_filter.cpp
  src/correlation_filter.cpp
  src/kalman.cpp
  src/pid.cpp
  src/gimbal.cpp
  src/tracker.cpp
  src/metrics.cpp
  src/synthetic.cpp
  src/overlay.cpp
  src/csv.cpp
  src/cli.cpp)
target_include_directories(vtt_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(vtt_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vtt_core PUBLIC opencv_core opencv_imgproc)
target_compile_definitions(vtt_core PUBLIC VTT_VERSION="${PROJECT_VERSION}")
vtt_set_project_options(vtt_core)

if(VTT_BUILD_APPS)
  add_executable(vtt apps/vtt.cpp)
  target_link_libraries(vtt PRIVATE vtt_core opencv_videoio opencv_highgui)
  vtt_set_project_options(vtt)

  add_executable(vtt_synth apps/vtt_synth.cpp)
  target_link_libraries(vtt_synth PRIVATE vtt_core opencv_videoio)
  vtt_set_project_options(vtt_synth)

  install(TARGETS vtt vtt_synth RUNTIME DESTINATION bin)
endif()

if(VTT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
