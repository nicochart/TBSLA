#ifndef TBSLA_CPP_MatrixELL
#define TBSLA_CPP_MatrixELL

#include <tbsla/cpp/Matrix.hpp>
#include <iostream>
#include <fstream>
#include <vector>

namespace tbsla { namespace cpp {

class MatrixELL : public virtual Matrix {
  public:
    MatrixELL() : values(0), columns(0), nnz(0), max_col(0) {};
    friend std::ostream & operator<<( std::ostream &os, const MatrixELL &m);
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const;
    std::vector<double> a_axpx_(const std::vector<double> &x, int vect_incr = 0) const;
    std::ostream & print_stats(std::ostream &os);
    std::ostream & print_infos(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1);
    int const get_nnz() {return nnz;};
    std::ostream& print(std::ostream& os) const;
    void fill_cdiag(int n_row, int n_col, int cdiag, int rp = 0, int RN = 1);
    void fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int rp = 0, int RN = 1);

  protected:
    std::vector<double> values;
    std::vector<int> columns;
    int nnz;
    int max_col;
};

}}

#endif
