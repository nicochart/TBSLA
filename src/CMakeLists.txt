cmake_minimum_required(VERSION 3.9...3.14)

project(TBSLA VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(cpp)

if(TBSLA_ENABLE_MPI)
    add_subdirectory(mpi)
endif()

if(TBSLA_ENABLE_HPX)
    add_subdirectory(hpx)
endif()
