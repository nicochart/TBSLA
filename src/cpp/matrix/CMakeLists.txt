list(APPEND SRC_LIST MatrixCSR.cpp MatrixCOO.cpp MatrixELL.cpp MatrixDENSE.cpp MatrixSCOO.cpp)

add_library(tbsla OBJECT ${SRC_LIST})
target_include_directories(tbsla PUBLIC ${PROJECT_SOURCE_DIR}/cpp/include)
