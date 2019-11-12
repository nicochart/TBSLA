#ifndef TBSLA_CPP_MatrixCSR
#define TBSLA_CPP_MatrixCSR

#include <tbsla/cpp/Matrix.hpp>
#include <iostream>
#include <vector>

class MatrixCSR : public Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const MatrixCSR &m);
    MatrixCSR(int n_row, int n_col, std::vector<double> & values, std::vector<int> & rowptr, std::vector<int> & colidx);
    std::vector<double> spmv(const std::vector<double> &v);
  protected:
    std::vector<double> values;
    std::vector<int> rowptr;
    std::vector<int> colidx;
};

#endif
