list(APPEND SRC_LIST MatrixCSR.cpp)
list(APPEND SRC_LIST MatrixCOO.cpp)
list(APPEND SRC_LIST MatrixELL.cpp)

message("list SRCLISMPI ${SRC_LIST}")

add_library(tbsla_mpi OBJECT ${SRC_LIST})
target_include_directories(tbsla_mpi PUBLIC ${PROJECT_SOURCE_DIR}/mpi/include)
target_link_libraries(tbsla_mpi PUBLIC MPI::MPI_CXX)
target_link_libraries(tbsla_mpi PUBLIC tbsla)
