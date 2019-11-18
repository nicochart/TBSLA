#ifndef TBSLA_CPP_Matrix
#define TBSLA_CPP_Matrix

#include <fstream>
#include <vector>

#ifdef TBSLA_HAS_MPI
#include <mpi.h>
#endif

class Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const Matrix &m) { return m.print(os); };
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0);
    std::vector<double> & saxpy(const std::vector<double> &x, std::vector<double> &y);

    int const get_n_row() {return n_row;}
    int const get_n_col() {return n_col;}
    int const get_nnz();

    virtual std::ostream & print_stats(std::ostream &os) = 0;
    virtual std::ostream & print_infos(std::ostream &os) = 0;
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is);
    virtual std::ostream& print(std::ostream& os) const = 0;

#ifdef TBSLA_HAS_MPI
    virtual int read_bin_mpiio(MPI_Comm comm, std::string filename) = 0;
    virtual std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0) = 0;
    int const get_gnnz() {return gnnz;};
#endif

  protected:
    int n_row, n_col;

#ifdef TBSLA_HAS_MPI
    int gnnz;
#endif
};

#endif
