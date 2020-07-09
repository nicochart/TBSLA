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
    int read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC);
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    void fill_cqmat(MPI_Comm comm, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
    std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    std::vector<double> a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    using tbsla::cpp::MatrixCSR::spmv;
    using tbsla::cpp::MatrixCSR::fill_cdiag;
    using tbsla::cpp::MatrixCSR::fill_cqmat;
    using tbsla::cpp::MatrixCSR::read;
    using tbsla::cpp::MatrixCSR::write;
};

}}

#endif
