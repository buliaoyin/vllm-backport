cmake_minimum_required(VERSION 3.26)
project(dsv41_backends LANGUAGES C CXX)

# Also usable in isolation for native backend development and correctness tests.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/external_projects/dsv41.cmake")
