#ifndef TBSLA_MPI_Matrix
#define TBSLA_MPI_Matrix

#include <tbsla/cpp/Matrix.hpp>

#include <fstream>
#include <vector>

#include <mpi.h>

namespace tbsla { namespace mpi {

class Matrix : public virtual tbsla::cpp::Matrix {
  public:
    virtual int read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) = 0;
    virtual std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0) = 0;
    virtual std::vector<double> a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0) = 0;
    std::vector<double> page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations);
    std::vector<double> personalized_page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, std::vector<int> personalized_nodes);
    int const get_gnnz() {return gnnz;};
    using tbsla::cpp::Matrix::fill_cdiag;
    using tbsla::cpp::Matrix::fill_cqmat;
    using tbsla::cpp::Matrix::read;
    using tbsla::cpp::Matrix::write;

  protected:
    int gnnz;
};

}}

#endif
