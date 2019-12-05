#ifndef TBSLA_CPP_MatrixCOO
#define TBSLA_CPP_MatrixCOO

#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <iostream>
#include <fstream>
#include <vector>

namespace tbsla { namespace cpp {

class MatrixCOO : public virtual Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const MatrixCOO &m);
    MatrixCOO(int n_row, int n_col, std::vector<double> & values, std::vector<int> & row, std::vector<int> & col);
    MatrixCOO(int n_row, int n_col, int n_values);
    MatrixCOO(int n_row, int n_col);
    MatrixCOO() : values(0), row(0), col(0) {};
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const;
    std::vector<double> a_axpx_(const std::vector<double> &x, int vect_incr = 0) const;
    void push_back(int r, int c, double v);
    std::ostream & print_stats(std::ostream &os);
    std::ostream & print_infos(std::ostream &os);
    std::ostream & print_as_dense(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1);
    int const get_nnz() {return values.size();};
    std::ostream& print(std::ostream& os) const;

    MatrixCSR toCSR();
    void fill_cdiag(int n_row, int n_col, int cdiag, int rp = 0, int RN = 1);

  protected:
    std::vector<double> values;
    std::vector<int> row;
    std::vector<int> col;
};

}}

#endif
