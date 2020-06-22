list(APPEND SRC_LIST MatrixCSR.cpp)
list(APPEND SRC_LIST MatrixCOO.cpp)
list(APPEND SRC_LIST MatrixSCOO.cpp)
list(APPEND SRC_LIST MatrixELL.cpp)
list(APPEND SRC_LIST MatrixDENSE.cpp)

message("list SRCLISMPI ${SRC_LIST}")

add_library(tbsla_mpi SHARED ${SRC_LIST})
target_include_directories(tbsla_mpi PUBLIC ${PROJECT_SOURCE_DIR}/mpi/include)
target_link_libraries(tbsla_mpi PUBLIC MPI::MPI_CXX)
target_link_libraries(tbsla_mpi PUBLIC tbsla)
target_include_directories(tbsla_mpi
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mpi/include>
  PUBLIC $<INSTALL_INTERFACE:include>
)
install(TARGETS tbsla_mpi)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/mpi/include/tbsla DESTINATION "include")
