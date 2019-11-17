add_library(MatrixUtils OBJECT mm.cpp InputParser.cpp)
target_include_directories(MatrixUtils PUBLIC ${PROJECT_SOURCE_DIR}/include)
