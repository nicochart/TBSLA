
add_executable(csr csr.cpp)
target_link_libraries(csr PRIVATE Matrix)
target_include_directories(csr PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS csr DESTINATION bin)

add_executable(coo coo.cpp)
target_link_libraries(coo PRIVATE Matrix)
target_include_directories(coo PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS coo DESTINATION bin)

add_executable(coo_to_csr coo_to_csr.cpp)
target_link_libraries(coo_to_csr PRIVATE Matrix)
target_include_directories(coo_to_csr PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS coo_to_csr DESTINATION bin)

add_executable(input_parser input_parser.cpp)
target_link_libraries(input_parser PRIVATE MatrixUtils)
target_link_libraries(input_parser PRIVATE Matrix)
target_include_directories(input_parser PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS input_parser DESTINATION bin)
