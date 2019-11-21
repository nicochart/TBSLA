#ifndef TBSLA_MPI_MatrixCSR
#define TBSLA_MPI_MatrixCSR

#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/mpi/Matrix.hpp>
#include <iostream>
#include <vector>

#include <mpi.h>

namespace tbsla { namespace mpi {

class MatrixCSR : public tbsla::cpp::MatrixCSR, public tbsla::mpi::Matrix {
  public:
    int read_bin_mpiio(MPI_Comm comm, std::string filename);
    std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    using tbsla::cpp::MatrixCSR::spmv;
  protected:
    int row_incr = 0; // index of the first value of the local array in the global array
};

}}

#endif
