
add_executable(csr csr.cpp)
target_link_libraries(csr PRIVATE tbsla)
add_test(csr_test csr)

add_executable(coo coo.cpp)
target_link_libraries(coo PRIVATE tbsla)
add_test(coo_test coo)

add_executable(coo_to_csr coo_to_csr.cpp)
target_link_libraries(coo_to_csr PRIVATE tbsla)

add_executable(input_parser input_parser.cpp)
target_link_libraries(input_parser PRIVATE tbsla)
add_test(input_parser_test input_parser)

add_executable(test_spmv_cdiag test_spmv_cdiag.cpp)
target_link_libraries(test_spmv_cdiag PRIVATE tbsla)
add_test(test_spmv_cdiag_test test_spmv_cdiag)

add_executable(test_a_axpx__cdiag test_a_axpx__cdiag.cpp)
target_link_libraries(test_a_axpx__cdiag PRIVATE tbsla)
add_test(test_a_axpx__cdiag_test test_a_axpx__cdiag)
