#ifndef TBSLA_CPP_MatrixCSR
#define TBSLA_CPP_MatrixCSR

#include <tbsla/cpp/Matrix.hpp>
#include <iostream>
#include <vector>

namespace tbsla { namespace cpp {

class MatrixCSR : public virtual Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const MatrixCSR &m);
    MatrixCSR(int n_row, int n_col, std::vector<double> & values, std::vector<int> & rowptr, std::vector<int> & colidx);
    MatrixCSR() : values(0), rowptr(0), colidx(0) {};
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const;
    std::vector<double> a_axpx_(const std::vector<double> &x, int vect_incr = 0) const;
    int const get_nnz();
    std::ostream & print_stats(std::ostream &os);
    std::ostream & print_infos(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1);
    std::ostream& print(std::ostream& os) const;
    void fill_cdiag(int n_row, int n_col, int cdiag, int rp = 0, int RN = 1);

  protected:
    std::vector<double> values;
    std::vector<int> rowptr;
    std::vector<int> colidx;
};

}}

#endif
