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
  if(this->nnz == 0)
    return r;
  for (int i = 0; i < std::min((size_t)this->n_row, this->values.size() / this->max_col); i++) {
    for (int j = 0; j < this->max_col; j++) {
      r[i + vect_incr] += this->values[i * this->max_col + j] * v[this->columns[i * this->max_col + j]];
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

void tbsla::cpp::MatrixELL::fill_cdiag(int n_row, int n_col, int cdiag, int rp, int RN) {
  this->n_row = n_row;
  this->n_col = n_col;

  this->values.clear();
  this->columns.clear();

  this->nnz = std::max(std::min(n_row, n_col - cdiag), 0) + std::max(std::min(n_row - cdiag, n_col), 0);
  if(this->nnz == 0)
    return;

  int s = tbsla::utils::range::pflv(this->n_row, rp, RN);
  int n = tbsla::utils::range::lnv(this->n_row, rp, RN);

  if(cdiag == 0) {
    this->nnz /= 2;
    this->values.reserve(n);
    this->columns.reserve(n);
    this->max_col = 1;
    for(int i = s; i < std::min(s + n, n_col); i++) {
      auto curr = tbsla::utils::values_generation::cdiag_value(i, nnz, n_row, n_col, cdiag);
      this->columns.push_back(std::get<1>(curr));
      this->values.push_back(std::get<2>(curr));
    }
  } else {
    this->values.reserve(2 * n);
    this->columns.reserve(2 * n);
    this->max_col = 2;

    int i;
    size_t incr = 0;

    if(s < std::min(cdiag, n_row)) {
      incr += std::max(std::min(n_col - cdiag, s), 0);
    } else if(s < std::min(n_row, n_col - cdiag)) {
      incr += std::max(std::min(n_col - cdiag, cdiag), 0);
      incr += (cdiag == 0 ? 1 : 2) * (s - std::min(n_col - cdiag, cdiag));
    } else {
      incr += std::max(std::min(n_col - cdiag, cdiag), 0);
      incr += (cdiag == 0 ? 1 : 2) * (std::max(n_col - 2 * cdiag, 0));
      incr += s - (n_col - cdiag) - (cdiag - std::min(n_col - cdiag, cdiag));
    }

    for(i = s; i < std::min( {cdiag, n_row, s + n} ); i++) {
      if(i < n_col - cdiag) {
        auto curr = tbsla::utils::values_generation::cdiag_value(incr, nnz, n_row, n_col, cdiag);
        this->columns.push_back(std::get<1>(curr));
        this->values.push_back(std::get<2>(curr));
        incr++;
      } else {
        this->columns.push_back(0);
        this->values.push_back(0);
      }
      this->columns.push_back(0);
      this->values.push_back(0);
    }
    for(; i < std::min( {n_row, n_col - cdiag, s + n} ); i++) {
      if(cdiag == 0) {
        auto curr = tbsla::utils::values_generation::cdiag_value(incr, nnz, n_row, n_col, cdiag);
        this->columns.push_back(std::get<1>(curr));
        this->values.push_back(std::get<2>(curr));
        incr++;
      } else {
        auto curr = tbsla::utils::values_generation::cdiag_value(incr, nnz, n_row, n_col, cdiag);
        this->columns.push_back(std::get<1>(curr));
        this->values.push_back(std::get<2>(curr));
        incr++;
        curr = tbsla::utils::values_generation::cdiag_value(incr, nnz, n_row, n_col, cdiag);
        this->columns.push_back(std::get<1>(curr));
        this->values.push_back(std::get<2>(curr));
        incr++;
      }
    }
    for(; i < std::min({n_row, s + n}); i++) {
      if(i < n_col + cdiag) {
        auto curr = tbsla::utils::values_generation::cdiag_value(incr, nnz, n_row, n_col, cdiag);
        this->columns.push_back(std::get<1>(curr));
        this->values.push_back(std::get<2>(curr));
        incr++;
      } else {
        this->columns.push_back(0);
        this->values.push_back(0);
      }
      this->columns.push_back(0);
      this->values.push_back(0);
    }
  }
}

void tbsla::cpp::MatrixELL::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int rp, int RN) {
  this->n_row = n_row;
  this->n_col = n_col;

  this->values.clear();
  this->columns.clear();

  int s = tbsla::utils::range::pflv(n_row, rp, RN);
  int n = tbsla::utils::range::lnv(n_row, rp, RN);

  int min_ = std::min(n_col - std::min(c, n_col) + 1, n_row);

  int incr = 0, nv = 0;
  for(int i = 0; i < min_; i++) {
    if(i < s) {
      incr += std::min(c, n_col);
    }
    if(i >= s && i < s + n) {
      nv += std::min(c, n_col);
    }
    if(i >= s + n) {
      break;
    }
  }
  for(int i = 0; i < std::min(n_row, n_col) - min_; i++) {
    if(i + min_ < s) {
      incr += std::min(c, n_col) - i - 1;
    }
    if(i + min_ >= s && i + min_ < s + n) {
      nv += std::min(c, n_col) - i - 1;
    }
    if(i + min_ >= s + n) {
      break;
    }
  }
  this->nnz = nv;
  this->max_col = std::min(c, n_col);

  if(nv == 0)
    return;

  this->values.reserve(std::min(c, n_col) * n);
  this->columns.reserve(std::min(c, n_col) * n);

  int i;
  for(i = s; i < std::min(min_, s + n); i++) {
    for(int j = 0; j < std::min(c, n_col); j++) {
      auto curr = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      this->columns.push_back(std::get<1>(curr));
      this->values.push_back(std::get<2>(curr));
      incr++;
    }
  }
  for(; i < std::min({n_row, s + n}); i++) {
    int j = 0;
    for(; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto curr = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      this->columns.push_back(std::get<1>(curr));
      this->values.push_back(std::get<2>(curr));
      incr++;
    }
    for(; j < std::min(c, n_col); j++) {
      this->columns.push_back(0);
      this->values.push_back(0);
    }
  }
}
