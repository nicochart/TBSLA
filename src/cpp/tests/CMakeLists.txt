
add_executable(csr csr.cpp)
target_link_libraries(csr PRIVATE MatrixCSR)
target_include_directories(csr PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS csr DESTINATION bin)
