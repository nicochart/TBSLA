list(APPEND SRC_LIST spmv.c)
list(APPEND SRC_LIST Matrix.c)
list(APPEND SRC_LIST Vector.c)

add_library(tbsla_parsec OBJECT ${SRC_LIST})
target_link_libraries(tbsla_parsec PUBLIC PaRSEC::parsec)
target_include_directories(tbsla_parsec PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(tbsla_parsec PUBLIC ${PROJECT_SOURCE_DIR}/parsec/include)
target_ptg_sources(tbsla_parsec PRIVATE spmv.jdf)
