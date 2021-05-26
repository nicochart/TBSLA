#ifndef TBSLA_CPP_MatrixCOO
#define TBSLA_CPP_MatrixCOO

#include <tbsla/cpp/Matrix.hpp>
#include <iostream>
#include <fstream>
#include <vector>

namespace tbsla { namespace cpp {

class MatrixCOO : public virtual Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const MatrixCOO &m);
    MatrixCOO(int n_row, int n_col, std::vector<double> & values, std::vector<int> & row, std::vector<int> & col);
    MatrixCOO(int n_row, int n_col, long int n_values);
    MatrixCOO(int n_row, int n_col);
    MatrixCOO() : values(0), row(0), col(0) {};
    const std::vector<double>& get_values() const { return values; }
    const std::vector<int>& get_row() const { return row; }
    const std::vector<int>& get_col() const { return col; }
    std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const;
    inline void Ax(std::vector<double> &r, const std::vector<double> &v, int vect_incr = 0) const;
    using tbsla::cpp::Matrix::a_axpx_;
    void push_back(int r, int c, double v);
    void update_nnz();
    std::ostream & print_stats(std::ostream &os);
    std::ostream & print_infos(std::ostream &os);
    std::ostream & print_as_dense(std::ostream &os);
    std::ostream & write(std::ostream &os);
    std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1);
    std::ostream& print(std::ostream& os) const;

    void readMM(std::string name);
    void fill_cdiag(int n_row, int n_col, int cdiag, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);
    void fill_cqmat_stochastic(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1);

    void normalize_columns();

  protected:
    std::vector<double> values;
    std::vector<int> row;
    std::vector<int> col;

  private:
    void init(int n_row, int n_col, long int n_values);
};

}}

#endif
