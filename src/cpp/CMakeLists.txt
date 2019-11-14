cmake_minimum_required(VERSION 3.9...3.14)

project(TBSLA VERSION 1.0 LANGUAGES CXX)

if(TBSLA_ENABLE_MPI)
    find_package(MPI REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(matrix)
add_subdirectory(utils)
add_subdirectory(apps)
add_subdirectory(tests)
