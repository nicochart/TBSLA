add_executable(tbsla_spmv_hpx tbsla_spmv_hpx.cpp)
target_link_libraries(tbsla_spmv_hpx PUBLIC tbsla_hpx)
target_link_libraries(tbsla_spmv_hpx PUBLIC tbsla)
install(TARGETS tbsla_spmv_hpx DESTINATION bin)

add_executable(tbsla_perf_hpx tbsla_perf_hpx.cpp)
target_link_libraries(tbsla_perf_hpx PUBLIC tbsla_hpx)
target_link_libraries(tbsla_perf_hpx PUBLIC tbsla)
install(TARGETS tbsla_perf_hpx DESTINATION bin)
