#ifndef TBSLA_MPI_Matrix
#define TBSLA_MPI_Matrix

#include <tbsla/cpp/Matrix.hpp>

#include <fstream>
#include <vector>

#include <mpi.h>

namespace tbsla { namespace mpi {

class Matrix : public virtual tbsla::cpp::Matrix {
  public:
    virtual int read_bin_mpiio(MPI_Comm comm, std::string filename) = 0;
    virtual std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0) = 0;
    int const get_gnnz() {return gnnz;};

  protected:
    int gnnz;
};

}}

#endif
