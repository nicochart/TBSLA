list(APPEND SRC_LIST MatrixCSR.cpp MatrixCOO.cpp)

if(TBSLA_ENABLE_MPI)
    list(APPEND SRC_LIST MatrixCOO_mpi.cpp)
    list(APPEND SRC_LIST MatrixCSR_mpi.cpp)
endif()

add_library(Matrix OBJECT ${SRC_LIST})
target_include_directories(Matrix PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(TBSLA_ENABLE_MPI)
    target_compile_definitions(Matrix PUBLIC TBSLA_HAS_MPI)
    target_link_libraries(Matrix PUBLIC MPI::MPI_CXX)
endif()
