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
  os << "n_row : " << this->n_row << std::endl;
  os << "n_col : " << this->n_col << std::endl;
  os << "n_values : " << this->nnz << std::endl;
  tbsla::utils::vector::streamvector<double>(os, "val", this->values);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "col", this->columns);
  os << std::endl;
  os << "-----------------" << std::endl;
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
  std::vector<double> r (this->n_row, 0);
  if(this->nnz == 0 || this->max_col == 0)
    return r;
  for (int i = 0; i < std::min((size_t)this->n_row, this->values.size() / this->max_col); i++) {
    for (int j = 0; j < this->max_col; j++) {
      r[i + this->f_row] += this->values[i * this->max_col + j] * v[this->columns[i * this->max_col + j]];
    }
  }
  return r;
}

std::vector<double> tbsla::cpp::MatrixELL::a_axpx_(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r = this->spmv(v, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  r = this->spmv(r, vect_incr);
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
  os.write(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));
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
  is.read(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));
  is.read(reinterpret_cast<char*>(&this->max_col), sizeof(this->max_col));


  size_t vec_size, depla_general, depla_local;
  depla_general = 4 * sizeof(int);


  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  depla_general += sizeof(size_t);

  return is;
}

void tbsla::cpp::MatrixELL::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;

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

  int nv = 0;
  for(int i = f_row; i < f_row + ln_row; i++) {
    int ii, jj;
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

  for(int i = f_row; i < f_row + ln_row; i++) {
    int ii, jj;
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

  this->values.clear();
  this->columns.clear();

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  int incr = 0, nv = 0;
  for(int i = 0; i < min_; i++) {
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
  for(int i = 0; i < std::min(n_row, n_col) - min_; i++) {
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
  int incr_save = incr;

  int i;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    int nbc = 0;
    for(int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      int ii, jj;
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
    int j = 0;
    for(; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      int ii, jj;
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
  int lincr;
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    lincr = 0;
    for(int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->columns.push_back(jj);
        this->values.push_back(std::get<2>(tuple));
        lincr++;
      }
      incr++;
    }
    for(int j = lincr; j < max_col; j++) {
      this->columns.push_back(0);
      this->values.push_back(0);
    }
  }
  for(; i < std::min({n_row, f_row + ln_row}); i++) {
    lincr = 0;
    for(int j = 0; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->columns.push_back(jj);
        this->values.push_back(std::get<2>(tuple));
        lincr++;
      }
      incr++;
    }
    for(int j = lincr; j < max_col; j++) {
      this->columns.push_back(0);
      this->values.push_back(0);
    }
  }
}
