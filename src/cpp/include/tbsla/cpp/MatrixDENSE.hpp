#ifndef TBSLA_CPP_MatrixDENSE
#define TBSLA_CPP_MatrixDENSE

#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/cpp/MatrixCOO.hpp>
#include <iostream>
#include <fstream>
#include <vector>

namespace tbsla { namespace cpp {

class MatrixDENSE : public virtual Matrix {
  public:
    MatrixDENSE() : values(0) {};
    MatrixDENSE(const tbsla::cpp::MatrixCOO & m);
    friend std::ostream & operator<<( std::ostream &os, const MatrixDENSE &m);
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const;
    using tbsla::cpp::Matrix::a_axpx_;
    std::ostream & print_stats(std::ostream &os);
    std::ostream & print_infos(std::ostream &os);
    std::ostream & print_as_dense(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1);
    std::ostream& print(std::ostream& os) const;
    void fill_cdiag(int n_row, int n_col, int cdiag, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_cqmat_stochastic(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);

    void normalize_columns();

  protected:
    std::vector<double> values;
};

}}

#endif
