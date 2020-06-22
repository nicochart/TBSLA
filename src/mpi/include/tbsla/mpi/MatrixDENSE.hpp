#ifndef TBSLA_MPI_MatrixDENSE
#define TBSLA_MPI_MatrixDENSE

#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/mpi/Matrix.hpp>
#include <iostream>
#include <vector>

#include <mpi.h>

namespace tbsla { namespace mpi {

class MatrixDENSE : public tbsla::cpp::MatrixDENSE, public tbsla::mpi::Matrix {
  public:
    int read_bin_mpiio(MPI_Comm comm, std::string filename);
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    void fill_cqmat(MPI_Comm comm, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
    std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    std::vector<double> a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    using tbsla::cpp::MatrixDENSE::spmv;
    using tbsla::cpp::MatrixDENSE::fill_cdiag;
    using tbsla::cpp::MatrixDENSE::fill_cqmat;
    using tbsla::cpp::MatrixDENSE::read;
    using tbsla::cpp::MatrixDENSE::write;
};

}}

#endif
