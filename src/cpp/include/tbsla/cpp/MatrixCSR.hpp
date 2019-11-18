#ifndef TBSLA_CPP_MatrixCSR
#define TBSLA_CPP_MatrixCSR

#include <tbsla/cpp/Matrix.hpp>
#include <iostream>
#include <vector>

class MatrixCSR : public Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const MatrixCSR &m);
    MatrixCSR(int n_row, int n_col, std::vector<double> & values, std::vector<int> & rowptr, std::vector<int> & colidx);
    MatrixCSR() : values(0), rowptr(0), colidx(0) {};
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0);
    int const get_nnz();
    std::ostream & print_stats(std::ostream &os);
    std::ostream & print_infos(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is);
    std::ostream& print(std::ostream& os) const;

#ifdef TBSLA_HAS_MPI
    int read_bin_mpiio(MPI_Comm comm, std::string filename);
    std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr = 0);
#endif
  protected:
    std::vector<double> values;
    std::vector<int> rowptr;
    std::vector<int> colidx;
#ifdef TBSLA_HAS_MPI
    int row_incr = 0; // index of the first value of the local array in the global array
#endif
};

#endif
