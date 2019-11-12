#ifndef TBSLA_CPP_MatrixCOO
#define TBSLA_CPP_MatrixCOO

#include <tbsla/cpp/Matrix.hpp>
#include <iostream>
#include <vector>

class MatrixCOO : public Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const MatrixCOO &m);
    MatrixCOO(int n_row, int n_col, std::vector<double> & values, std::vector<int> & row, std::vector<int> & col);
    std::vector<double> spmv(const std::vector<double> &v);
  protected:
    std::vector<double> values;
    std::vector<int> row;
    std::vector<int> col;
};

#endif
