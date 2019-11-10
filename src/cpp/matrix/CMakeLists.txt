add_library(Matrix OBJECT MatrixCSR.cpp MatrixCOO.cpp)
target_include_directories(Matrix PUBLIC ${PROJECT_SOURCE_DIR}/include)
