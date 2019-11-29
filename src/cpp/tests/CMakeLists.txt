
add_executable(csr csr.cpp)
target_link_libraries(csr PRIVATE tbsla)
install(TARGETS csr DESTINATION bin)
add_test(csr_test csr)

add_executable(coo coo.cpp)
target_link_libraries(coo PRIVATE tbsla)
install(TARGETS coo DESTINATION bin)
add_test(coo_test coo)

add_executable(coo_to_csr coo_to_csr.cpp)
target_link_libraries(coo_to_csr PRIVATE tbsla)
install(TARGETS coo_to_csr DESTINATION bin)

add_executable(input_parser input_parser.cpp)
target_link_libraries(input_parser PRIVATE tbsla)
install(TARGETS input_parser DESTINATION bin)
add_test(input_parser_test input_parser)

add_executable(test_spmv_cdiag test_spmv_cdiag.cpp)
target_link_libraries(test_spmv_cdiag PRIVATE tbsla)
install(TARGETS test_spmv_cdiag DESTINATION bin)
add_test(test_spmv_cdiag_test test_spmv_cdiag)
