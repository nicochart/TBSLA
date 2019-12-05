#ifndef TBSLA_MPI_MatrixELL
#define TBSLA_MPI_MatrixELL

#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/mpi/Matrix.hpp>
#include <iostream>
#include <vector>

#include <mpi.h>

namespace tbsla { namespace mpi {

class MatrixELL : public tbsla::cpp::MatrixELL, public tbsla::mpi::Matrix {
  public:
    int read_bin_mpiio(MPI_Comm comm, std::string filename);
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    using tbsla::cpp::MatrixELL::spmv;
  protected:
    int row_incr = 0; // index of the first value of the local array in the global array
};

}}

#endif
