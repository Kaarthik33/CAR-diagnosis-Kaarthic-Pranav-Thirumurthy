cmake_minimum_required(VERSION 3.16)
project(voltsense VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

add_library(voltsense_core cpp/src/trip_features.cpp)
target_include_directories(voltsense_core PUBLIC cpp/include)

add_executable(voltsense_extract cpp/apps/extract_features.cpp)
target_link_libraries(voltsense_extract PRIVATE voltsense_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(voltsense_tests cpp/tests/test_core.cpp)
  target_link_libraries(voltsense_tests PRIVATE voltsense_core)
  add_test(NAME core COMMAND voltsense_tests)
endif()
