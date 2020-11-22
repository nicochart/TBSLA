#ifndef TBSLA_CPP_MatrixSCOO
#define TBSLA_CPP_MatrixSCOO

#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <iostream>
#include <fstream>
#include <vector>

namespace tbsla { namespace cpp {

class MatrixSCOO : public virtual Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const MatrixSCOO &m);
    MatrixSCOO(int n_row, int n_col, std::vector<double> & values, std::vector<int> & row, std::vector<int> & col);
    MatrixSCOO(int n_row, int n_col, int n_values);
    MatrixSCOO(int n_row, int n_col);
    MatrixSCOO() : values(0), row(0), col(0) {};
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const;
    using tbsla::cpp::Matrix::a_axpx_;
    void push_back(int r, int c, double v);
    std::ostream & print_stats(std::ostream &os);
    std::ostream & print_infos(std::ostream &os);
    std::ostream & print_as_dense(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1);
    std::ostream& print(std::ostream& os) const;

    MatrixCSR toCSR();
    void fill_cdiag(int n_row, int n_col, int cdiag, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_cqmat_stochastic(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);

    void normalize_columns();

  protected:
    std::vector<double> values;
    std::vector<int> row;
    std::vector<int> col;
};

}}

#endif
