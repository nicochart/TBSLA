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
    virtual std::vector<double> page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, int &nb_iterations_done);
    virtual std::vector<double> personalized_page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, std::vector<int> personalized_nodes, int &nb_iterations_done);
    int const get_gnnz() {return gnnz;};
    int const compute_sum_nnz(MPI_Comm comm);
    int const compute_min_nnz(MPI_Comm comm);
    int const compute_max_nnz(MPI_Comm comm);
    using tbsla::cpp::Matrix::fill_cdiag;
    using tbsla::cpp::Matrix::fill_cqmat;
    using tbsla::cpp::Matrix::read;
    using tbsla::cpp::Matrix::write;

  protected:
    int gnnz;
};

}}

#endif
