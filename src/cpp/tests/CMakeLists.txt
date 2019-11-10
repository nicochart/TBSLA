
add_executable(csr csr.cpp)
target_link_libraries(csr PRIVATE Matrix)
target_include_directories(csr PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS csr DESTINATION bin)

add_executable(coo coo.cpp)
target_link_libraries(coo PRIVATE Matrix)
target_include_directories(coo PUBLIC ${PROJECT_SOURCE_DIR}/include)
install(TARGETS coo DESTINATION bin)
