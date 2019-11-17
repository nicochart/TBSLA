#ifndef TBSLA_CPP_Matrix
#define TBSLA_CPP_Matrix

#include <fstream>
#include <vector>

class Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const Matrix &m) { return m.print(os); };
    std::vector<double> spmv(const std::vector<double> &v);
    std::vector<double> & saxpy(const std::vector<double> &x, std::vector<double> &y);

    int const get_n_row() {return n_row;}
    int const get_n_col() {return n_col;}
    int const get_nnz();
    virtual std::ostream& print(std::ostream& os) const = 0;
  protected:
    int n_row, n_col;
};

#endif
