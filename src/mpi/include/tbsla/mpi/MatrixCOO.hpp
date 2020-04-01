#ifndef TBSLA_MPI_MatrixCOO
#define TBSLA_MPI_MatrixCOO

#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/cpp/MatrixCOO.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#include <mpi.h>

namespace tbsla { namespace mpi {

class MatrixCOO : public tbsla::cpp::MatrixCOO, public virtual tbsla::mpi::Matrix {
  public:
    int read_bin_mpiio(MPI_Comm comm, std::string filename);
    void fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag);
    void fill_cqmat(MPI_Comm comm, int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC);
    std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    std::vector<double> a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
    using tbsla::cpp::MatrixCOO::spmv;
    using tbsla::cpp::MatrixCOO::fill_cdiag;
    using tbsla::cpp::MatrixCOO::fill_cqmat;
};

}}

#endif
