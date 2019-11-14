#ifndef TBSLA_CPP_MatrixCOO
#define TBSLA_CPP_MatrixCOO

#include <tbsla/cpp/Matrix.hpp>
#include <iostream>
#include <fstream>
#include <vector>

#ifdef TBSLA_HAS_MPI
#include <mpi.h>
#endif

class MatrixCOO : public Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const MatrixCOO &m);
    MatrixCOO(int n_row, int n_col, std::vector<double> & values, std::vector<int> & row, std::vector<int> & col);
    MatrixCOO(int n_row, int n_col, int n_values);
    MatrixCOO(int n_row, int n_col);
    MatrixCOO() : values(0), row(0), col(0) {};
    std::vector<double> spmv(const std::vector<double> &v);
    void push_back(int r, int c, double v);
    std::ostream & print_stats(std::ostream &os);
    std::ostream & print_infos(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is);
    int const get_nnz() {return values.size();};

#ifdef TBSLA_HAS_MPI
   int read_bin_mpiio(MPI_Comm comm, std::string filename);
    std::vector<double> spmv(MPI_Comm comm, const std::vector<double> &v);
#endif
  protected:
    std::vector<double> values;
    std::vector<int> row;
    std::vector<int> col;
};

#endif
