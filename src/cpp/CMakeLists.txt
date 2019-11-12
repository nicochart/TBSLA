cmake_minimum_required(VERSION 3.8.2...3.14)

project(TBSLA VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(matrix)
add_subdirectory(utils)
add_subdirectory(apps)
add_subdirectory(tests)
