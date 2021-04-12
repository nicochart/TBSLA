list(APPEND SRC_LIST MatrixCSR.cpp MatrixCOO.cpp MatrixELL.cpp MatrixDENSE.cpp MatrixSCOO.cpp Matrix.cpp)

add_library(tbsla SHARED ${SRC_LIST})
target_include_directories(tbsla
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>
  PUBLIC $<INSTALL_INTERFACE:include>
)
target_link_libraries(tbsla PUBLIC tbsla_utils)
install(TARGETS tbsla)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/cpp/include/tbsla DESTINATION "include")
