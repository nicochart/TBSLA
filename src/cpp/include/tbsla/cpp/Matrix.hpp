#ifndef TBSLA_CPP_Matrix
#define TBSLA_CPP_Matrix

#include <fstream>
#include <vector>

namespace tbsla { namespace cpp {

class Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const Matrix &m) { return m.print(os); };
    virtual std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const = 0;
    std::vector<double> a_axpx_(const std::vector<double> &v, int vect_incr = 0) const;
    std::vector<double> & saxpy(const std::vector<double> &x, std::vector<double> &y);
    std::vector<double> page_rank(double epsilon, double beta, int max_iteration, int& nb_iterations_done);
    std::vector<double> personalized_page_rank(double epsilon, double beta, int max_iteration, std::vector<int> personalized_nodes, int& nb_iterations_done);
    int const get_n_row() {return n_row;}
    int const get_n_col() {return n_col;}
    int const get_f_row() {return f_row;}
    int const get_f_col() {return f_col;}
    int const get_ln_row() {return ln_row;}
    int const get_ln_col() {return ln_col;}
    long int const get_nnz() {return nnz;};

    virtual std::ostream & print_stats(std::ostream &os) = 0;
    virtual std::ostream & print_infos(std::ostream &os) = 0;
    virtual std::ostream & write(std::ostream &os) = 0;
    virtual std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1) = 0;
    virtual std::ostream& print(std::ostream& os) const = 0;
    virtual std::ostream& print_as_dense(std::ostream& os) = 0;

    virtual void fill_cdiag(int n_row, int n_col, int cdiag, int pr = 0, int pc = 0, int NR = 1, int NC = 1) = 0;
    virtual void fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1) = 0;
    virtual void fill_cqmat_stochastic(int n_row, int n_col, int c, double q, unsigned int seed_mult = 1, int pr = 0, int pc = 0, int NR = 1, int NC = 1) = 0;

    virtual void normalize_columns() = 0;

  protected:
    int n_row, n_col, f_row, f_col, ln_row, ln_col, pr, pc, NR, NC;
    long int nnz;

};

}}

#endif
