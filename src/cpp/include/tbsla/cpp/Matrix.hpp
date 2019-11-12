#ifndef TBSLA_CPP_Matrix
#define TBSLA_CPP_Matrix

#include <vector>

class Matrix {
  public:
    std::vector<double> spmv(const std::vector<double> &v);
    std::vector<double> & saxpy(const std::vector<double> &x, std::vector<double> &y);
  protected:
    int n_row, n_col;
};

#endif
