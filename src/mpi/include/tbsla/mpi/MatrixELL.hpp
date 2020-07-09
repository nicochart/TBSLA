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
    int read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC);
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    void fill_cqmat(MPI_Comm comm, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
    std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    std::vector<double> a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    using tbsla::cpp::MatrixELL::spmv;
    using tbsla::cpp::MatrixELL::fill_cdiag;
    using tbsla::cpp::MatrixELL::fill_cqmat;
    using tbsla::cpp::MatrixELL::read;
    using tbsla::cpp::MatrixELL::write;
};

}}

#endif
