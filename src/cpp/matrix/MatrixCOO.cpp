#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/csr.hpp>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col, std::vector<double> & values, std::vector<int> & row,  std::vector<int> & col) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->values = values;
  this->row = row;
  this->col = col;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
}

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col, int n_values) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->values.reserve(n_values);
  this->row.reserve(n_values);
  this->col.reserve(n_values);
}

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col) {
  this->n_row = n_row;
  this->n_col = n_col;
}

std::ostream& tbsla::cpp::MatrixCOO::print_as_dense(std::ostream& os) {
  std::vector<double> d(this->n_row * this->n_col, 0);
  for(int i = 0; i < this->row.size(); i++) {
    d[row[i] * this->n_col + col[i]] += this->values[i];
  }
  tbsla::utils::vector::print_dense_matrix(this->n_row, this->n_col, d, os);
  return os;
}

std::ostream& tbsla::cpp::MatrixCOO::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "------ COO ------" << std::endl;
  os << "-----------------" << std::endl;
  os << "number of rows             -  n_row   : " << this->n_row << std::endl;
  os << "number of columns          -  n_col   : " << this->n_col << std::endl;
  os << "local number of rows       -  ln_row  : " << this->ln_row << std::endl;
  os << "local number of columns    -  ln_col  : " << this->ln_col << std::endl;
  os << "first row                  -  f_row   : " << this->f_row << std::endl;
  os << "first column               -  f_col   : " << this->f_col << std::endl;
  os << "number of non-zero elts    -  nnz     : " << this->nnz << std::endl;
  os << "block position (row)       -  pr      : " << this->pr << std::endl;
  os << "block position (column)    -  pc      : " << this->pc << std::endl;
  os << "number of blocks (row)     -  NR      : " << this->NR << std::endl;
  os << "number of blocks (column)  -  NC      : " << this->NC << std::endl;
  tbsla::utils::vector::streamvector<double>(os, "val", this->values);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "row", this->row);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "col", this->col);
  os << std::endl;
  os << "-----------------" << std::endl << std::flush;
  return os;
}

std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixCOO &m) {
  return m.print(os);
}

std::vector<double> tbsla::cpp::MatrixCOO::spmv(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r (this->n_row, 0);
  for (int i = 0; i < this->values.size(); i++) {
     r[this->row[i] + vect_incr] += this->values[i] * v[this->col[i]];
  }
  return r;
}

void tbsla::cpp::MatrixCOO::push_back(int r, int c, double v) {
  if(r >= this->n_row || r < 0)
    return;
  if(c >= this->n_col || c < 0)
    return;
  this->values.push_back(v);
  this->row.push_back(r);
  this->col.push_back(c);
}

std::ostream & tbsla::cpp::MatrixCOO::print_infos(std::ostream &os) {
  os << "-----------------" << std::endl;
  os << "------ COO ------" << std::endl;
  os << "--- general   ---" << std::endl;
  os << "n_row : " << n_row << std::endl;
  os << "n_col : " << n_col << std::endl;
  os << "--- capacity  ---" << std::endl;
  os << "values : " << values.capacity() << std::endl;
  os << "row : " << row.capacity() << std::endl;
  os << "col : " << col.capacity() << std::endl;
  os << "--- size      ---" << std::endl;
  os << "values : " << values.size() << std::endl;
  os << "row : " << row.size() << std::endl;
  os << "col : " << col.size() << std::endl;
  os << "-----------------" << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixCOO::print_stats(std::ostream &os) {
  int s = 0, u = 0, d = 0;
  if(row.size() != col.size()) {
    std::cerr << "Err \n";
    return os;
  }
  for(int i = 0; i < row.size(); i++) {
    if(row[i] < col[i]) {
      s++;
    } else if(row[i] > col[i]) {
      u++;
    } else {
      d++;
    }
  }
  os << "upper values : " << u << std::endl;
  os << "lower values : " << s << std::endl;
  os << "diag  values : " << d << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixCOO::write(std::ostream &os) {
  os.write(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  os.write(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));
  os.write(reinterpret_cast<char*>(&this->ln_row), sizeof(this->ln_row));
  os.write(reinterpret_cast<char*>(&this->ln_col), sizeof(this->ln_col));
  os.write(reinterpret_cast<char*>(&this->f_row), sizeof(this->f_row));
  os.write(reinterpret_cast<char*>(&this->f_col), sizeof(this->f_col));
  os.write(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));
  os.write(reinterpret_cast<char*>(&this->pr), sizeof(this->pr));
  os.write(reinterpret_cast<char*>(&this->pc), sizeof(this->pc));
  os.write(reinterpret_cast<char*>(&this->NR), sizeof(this->NR));
  os.write(reinterpret_cast<char*>(&this->NC), sizeof(this->NC));

  size_t size_v = this->values.size();
  os.write(reinterpret_cast<char*>(&size_v), sizeof(size_v));
  os.write(reinterpret_cast<char*>(this->values.data()), this->values.size() * sizeof(double));

  size_t size_r = this->row.size();
  os.write(reinterpret_cast<char*>(&size_r), sizeof(size_r));
  os.write(reinterpret_cast<char*>(this->row.data()), this->row.size() * sizeof(int));

  size_t size_c = this->col.size();
  os.write(reinterpret_cast<char*>(&size_c), sizeof(size_c));
  os.write(reinterpret_cast<char*>(this->col.data()), this->col.size() * sizeof(int));
  return os;
}

std::istream & tbsla::cpp::MatrixCOO::read(std::istream &is, std::size_t pos, std::size_t n) {
  is.read(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  is.read(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));
  is.read(reinterpret_cast<char*>(&this->ln_row), sizeof(this->ln_row));
  is.read(reinterpret_cast<char*>(&this->ln_col), sizeof(this->ln_col));
  is.read(reinterpret_cast<char*>(&this->f_row), sizeof(this->f_row));
  is.read(reinterpret_cast<char*>(&this->f_col), sizeof(this->f_col));
  is.read(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));
  is.read(reinterpret_cast<char*>(&this->pr), sizeof(this->pr));
  is.read(reinterpret_cast<char*>(&this->pc), sizeof(this->pc));
  is.read(reinterpret_cast<char*>(&this->NR), sizeof(this->NR));
  is.read(reinterpret_cast<char*>(&this->NC), sizeof(this->NC));


  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->values.resize(size);
  is.read(reinterpret_cast<char*>(this->values.data()), size * sizeof(double));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->row.resize(size);
  is.read(reinterpret_cast<char*>(this->row.data()), size * sizeof(int));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->col.resize(size);
  is.read(reinterpret_cast<char*>(this->col.data()), size * sizeof(int));
  return is;
}

tbsla::cpp::MatrixCSR tbsla::cpp::MatrixCOO::toCSR() {
  std::vector<int> p(this->values.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(), [&](unsigned i, unsigned j){ return tbsla::cpp::utils::csr::compare_row(this->row, this->col, i, j); });

  std::vector<int> pr = tbsla::cpp::utils::csr::applyPermutation<int>(p, this->row);
  std::vector<int> pc = tbsla::cpp::utils::csr::applyPermutation<int>(p, this->col);
  std::vector<double> pv = tbsla::cpp::utils::csr::applyPermutation<double>(p, this->values);

  if(this->values.size() == 0) {
    std::vector<int> cr(this->n_row + 1, 0);
    return tbsla::cpp::MatrixCSR(this->n_row, this->n_col, pv, cr, pc);
  }

  std::vector<int> cr(this->n_row + 1);
  cr[0] = 0;
  size_t incr = 1;
  for(int i = 0; i < pr[0]; i++) {
    incr++;
    cr[incr] = 0;
  }
  for(int i = 0; i < pr.size() - 1; i++) {
    cr[incr]++;
    if(pr[i] != pr[i + 1]) {
      for(int j = 0; j < pr[i + 1] - pr[i]; j++) {
        incr++;
        cr[incr] = cr[incr - 1];
      }
    }
  }
  cr[incr]++;
  for(int i = incr; i < this->n_row; i++) {
    cr[i + 1] = cr[i];
  }
  return tbsla::cpp::MatrixCSR(this->n_row, this->n_col, pv, cr, pc);
}

void tbsla::cpp::MatrixCOO::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->values.clear();
  this->col.clear();
  this->row.clear();

  this->ln_row = this->NR;
  this->ln_col = this->NC;
  this->f_row = 0;
  this->f_col = 0;

  int gnv = std::max(std::min(n_row, n_col - cdiag), 0) + std::max(std::min(n_row - cdiag, n_col), 0);
  if(cdiag == 0)
    gnv /= 2;
  this->nnz = 0;
  if(gnv == 0)
    return;

  int s = tbsla::utils::range::pflv(gnv, pr * NC + pc, NR * NC);
  int n = tbsla::utils::range::lnv(gnv, pr * NC + pc, NR * NC);
  this->nnz = n;

  this->values.reserve(n);
  this->col.reserve(n);
  this->row.reserve(n);

  for(int i = s; i < s + n; i++) {
    auto tuple = tbsla::utils::values_generation::cdiag_value(i, gnv, n_row, n_col, cdiag);
    this->push_back(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
  }
}

void tbsla::cpp::MatrixCOO::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->values.clear();
  this->col.clear();
  this->row.clear();

  this->ln_row = this->NR;
  this->ln_col = this->NC;
  this->f_row = 0;
  this->f_col = 0;

  int gnv = 0;
  for(int i = 0; i < std::min(n_col - std::min(c, n_col) + 1, n_row); i++) {
    gnv += std::min(c, n_col);
  }
  for(int i = 0; i < std::min(n_row, n_col) - std::min(n_col - std::min(c, n_col) + 1, n_row); i++) {
    gnv += std::min(c, n_col) - i - 1;
  }
  if(gnv == 0)
    return;

  int s = tbsla::utils::range::pflv(gnv, pr * NC + pc, NR * NC);
  int n = tbsla::utils::range::lnv(gnv, pr * NC + pc, NR * NC);
  this->nnz = n;

  this->values.reserve(n);
  this->col.reserve(n);
  this->row.reserve(n);

  for(int i = s; i < s + n; i++) {
    auto tuple = tbsla::utils::values_generation::cqmat_value(i, n_row, n_col, c, q, seed_mult);
    this->push_back(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
  }

  this->values.shrink_to_fit();
  this->col.shrink_to_fit();
  this->row.shrink_to_fit();
}
