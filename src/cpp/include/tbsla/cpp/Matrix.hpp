#ifndef TBSLA_CPP_Matrix
#define TBSLA_CPP_Matrix

#include <fstream>
#include <vector>

namespace tbsla { namespace cpp {

class Matrix {
  public:
    friend std::ostream & operator<<( std::ostream &os, const Matrix &m) { return m.print(os); };
    virtual std::vector<double> spmv(const std::vector<double> &v, int vect_incr = 0) const = 0;
    std::vector<double> & saxpy(const std::vector<double> &x, std::vector<double> &y);

    int const get_n_row() {return n_row;}
    int const get_n_col() {return n_col;}
    int const get_nnz();

    virtual std::ostream & print_stats(std::ostream &os) = 0;
    virtual std::ostream & print_infos(std::ostream &os) = 0;
    virtual std::ostream & write(std::ostream &os) = 0;
    virtual std::istream & read(std::istream &is, std::size_t pos = 0, std::size_t n = 1) = 0;
    virtual std::ostream& print(std::ostream& os) const = 0;

  protected:
    int n_row, n_col;

};

}}

#endif
