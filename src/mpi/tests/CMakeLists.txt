add_executable(mpi_test_spmv_cdiag test_spmv_cdiag.cpp)
target_link_libraries(mpi_test_spmv_cdiag PRIVATE tbsla_mpi)
# modern cmake (3.9+)
# https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
target_link_libraries(mpi_test_spmv_cdiag PUBLIC MPI::MPI_CXX)
target_link_libraries(mpi_test_spmv_cdiag PUBLIC tbsla)
add_test (mpi_test_spmv_cdiag_test ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} mpi_test_spmv_cdiag ${MPIEXEC_POSTFLAGS})

