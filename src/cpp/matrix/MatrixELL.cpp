#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

std::ostream& tbsla::cpp::MatrixELL::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "------ ELL ------" << std::endl;
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
  os << "maximum number of columns  -  max_col : " << this->max_col << std::endl;
  tbsla::utils::vector::streamvector<double>(os, "val", this->values);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "col", this->columns);
  os << std::endl;
  os << "-----------------" << std::endl << std::flush;
  return os;
}

std::ostream& tbsla::cpp::MatrixELL::print_as_dense(std::ostream& os) {
  std::vector<double> d(this->n_row * this->n_col, 0);
  for (int i = 0; i < std::min((size_t)this->n_row, this->values.size() / this->max_col); i++) {
    for (int j = 0; j < this->max_col; j++) {
      d[i * this->n_col + this->columns[i * this->max_col + j]] += this->values[i * this->max_col + j];
    }
  }
  tbsla::utils::vector::print_dense_matrix(this->n_row, this->n_col, d, os);
  return os;
}

std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixELL &m) {
  return m.print(os);
}

std::vector<double> tbsla::cpp::MatrixELL::spmv(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r (this->ln_row, 0);
  if(this->nnz == 0 || this->max_col == 0)
    return r;
  #pragma omp parallel for
  for (int i = 0; i < std::min((size_t)this->ln_row, this->values.size() / this->max_col); i++) {
    for (int j = 0; j < this->max_col; j++) {
      int idx = this->columns[i * this->max_col + j] - this->f_col;
      if(idx < 0) idx = 0;
      r[i] += this->values[i * this->max_col + j] * v[idx];
    }
  }
  return r;
}

std::ostream & tbsla::cpp::MatrixELL::print_infos(std::ostream &os) {
  os << "-----------------" << std::endl;
  os << "------ ELL ------" << std::endl;
  os << "--- general   ---" << std::endl;
  os << "n_row : " << n_row << std::endl;
  os << "n_col : " << n_col << std::endl;
  os << "nnz : " << nnz << std::endl;
  os << "max_col : " << max_col << std::endl;
  os << "--- capacity  ---" << std::endl;
  os << "values : " << values.capacity() << std::endl;
  os << "columns : " << columns.capacity() << std::endl;
  os << "--- size      ---" << std::endl;
  os << "values : " << values.size() << std::endl;
  os << "columns : " << columns.size() << std::endl;
  os << "-----------------" << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixELL::print_stats(std::ostream &os) {
  int s = 0, u = 0, d = 0;
  os << "upper values : " << u << std::endl;
  os << "lower values : " << s << std::endl;
  os << "diag  values : " << d << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixELL::write(std::ostream &os) {
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
  os.write(reinterpret_cast<char*>(&this->max_col), sizeof(this->max_col));

  size_t size_v = this->values.size();
  os.write(reinterpret_cast<char*>(&size_v), sizeof(size_v));
  os.write(reinterpret_cast<char*>(this->values.data()), this->values.size() * sizeof(double));

  size_t size_c = this->columns.size();
  os.write(reinterpret_cast<char*>(&size_c), sizeof(size_c));
  os.write(reinterpret_cast<char*>(this->columns.data()), this->columns.size() * sizeof(int));
  return os;
}

std::istream & tbsla::cpp::MatrixELL::read(std::istream &is, std::size_t pos, std::size_t n) {
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
  is.read(reinterpret_cast<char*>(&this->max_col), sizeof(this->max_col));


  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->values.resize(size);
  is.read(reinterpret_cast<char*>(this->values.data()), size * sizeof(double));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->columns.resize(size);
  is.read(reinterpret_cast<char*>(this->columns.data()), size * sizeof(int));
  return is;
}

void tbsla::cpp::MatrixELL::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->values.clear();
  this->columns.clear();

  ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  if(cdiag == 0)
    this->max_col = 1;
  else
    this->max_col = 2;

  long int nv = 0;
  for(long int i = f_row; i < f_row + ln_row; i++) {
    long int ii, jj;
    jj = i - cdiag;
    ii = i;
    if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
      nv++;
    }
    if(cdiag != 0) {
      jj = i + cdiag;
      ii = i;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        nv++;
      }
    }
  }

  if(nv == 0)
    return;

  this->nnz = nv;
  this->values.reserve(this->max_col * this->ln_row);
  this->columns.reserve(this->max_col * this->ln_row);

  for(long int i = f_row; i < f_row + ln_row; i++) {
    long int ii, jj;
    jj = i - cdiag;
    ii = i;
    if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
      this->columns.push_back(jj);
      this->values.push_back(1);
    } else {
      this->columns.push_back(0);
      this->values.push_back(0);
    }
    if(cdiag != 0) {
      jj = i + cdiag;
      ii = i;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->columns.push_back(jj);
        this->values.push_back(1);
      } else {
        this->columns.push_back(0);
        this->values.push_back(0);
      }
    }
  }
}

void tbsla::cpp::MatrixELL::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->values.clear();
  this->columns.clear();

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  long int incr = 0, nv = 0;
  for(long int i = 0; i < min_; i++) {
    if(i < f_row) {
      incr += std::min(c, n_col);
    }
    if(i >= f_row && i < f_row + ln_row) {
      nv += std::min(c, n_col);
    }
    if(i >= f_row + ln_row) {
      break;
    }
  }
  for(long int i = 0; i < std::min(n_row, n_col) - min_; i++) {
    if(i + min_ < f_row) {
      incr += std::min(c, n_col) - i - 1;
    }
    if(i + min_ >= f_row && i + min_ < f_row + ln_row) {
      nv += std::min(c, n_col) - i - 1;
    }
    if(i + min_ >= f_row + ln_row) {
      break;
    }
  }

  this->nnz = 0;
  this->max_col = 0;
  long int incr_save = incr;

  long int i;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    int nbc = 0;
    for(long int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->nnz++;
        nbc++;
      }
      incr++;
    }
    this->max_col = std::max(this->max_col, nbc);
  }
  for(; i < std::min({n_row, n_col, f_row + ln_row}); i++) {
    int nbc = 0;
    long int j = 0;
    for(; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        nbc++;
        this->nnz++;
      }
      incr++;
    }
    for(; j < std::min(c, n_col); j++) {
      if(j >= f_col && j < f_col + ln_col) {
        nbc++;
      }
    }
    this->max_col = std::max(this->max_col, nbc);
  }

  if(nv == 0)
    return;

  this->values.reserve(this->max_col * ln_row);
  this->columns.reserve(this->max_col * ln_row);

  incr = incr_save;
  long int lincr;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    lincr = 0;
    for(long int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->columns.push_back(jj);
        this->values.push_back(std::get<2>(tuple));
        lincr++;
      }
      incr++;
    }
    for(long int j = lincr; j < max_col; j++) {
      this->columns.push_back(0);
      this->values.push_back(0);
    }
  }
  for(; i < std::min({n_row, f_row + ln_row}); i++) {
    lincr = 0;
    for(long int j = 0; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      long int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->columns.push_back(jj);
        this->values.push_back(std::get<2>(tuple));
        lincr++;
      }
      incr++;
    }
    for(long int j = lincr; j < max_col; j++) {
      this->columns.push_back(0);
      this->values.push_back(0);
    }
  }
}

void tbsla::cpp::MatrixELL::fill_cqmat_stochastic(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->fill_cqmat(n_row, n_col, c, q, seed_mult, pr, pc, NR, NC);
  if(this->values.size() == 0)
    return;
  std::vector<double> sum = tbsla::utils::values_generation::cqmat_sum_columns(n_row, n_col, c, q, seed_mult);
  for (long int i = 0; i < std::min((size_t)this->ln_row, this->values.size() / this->max_col); i++) {
    for (long int j = 0; j < this->max_col; j++) {
      if(this->values[i * this->max_col + j] != 0) this->values[i * this->max_col + j] /= sum[this->columns[i * this->max_col + j]];
    }
  }
}

void tbsla::cpp::MatrixELL::normalize_columns() {
  std::vector<double> sum(this->ln_col, 0);
  for (long int i = 0; i < std::min((size_t)this->ln_row, this->values.size() / this->max_col); i++) {
    for (long int j = 0; j < this->max_col; j++) {
      long int idx = this->columns[i * this->max_col + j] - this->f_col;
      if(idx < 0) idx = 0;
      sum[idx] += this->values[i * this->max_col + j];
    }
  }
  for (long int i = 0; i < std::min((size_t)this->ln_row, this->values.size() / this->max_col); i++) {
    for (long int j = 0; j < this->max_col; j++) {
      long int idx = this->columns[i * this->max_col + j] - this->f_col;
      if(idx < 0) idx = 0;
      this->values[i * this->max_col + j] /= sum[idx];
    }
  }
}
