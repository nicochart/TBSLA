#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

tbsla::cpp::MatrixCSR::MatrixCSR(int n_row, int n_col, std::vector<double> & values, std::vector<int> & rowptr,  std::vector<int> & colidx) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->values = values;
  this->rowptr = rowptr;
  this->colidx = colidx;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
}

std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixCSR &m) {
  return m.print(os);
}

std::ostream& tbsla::cpp::MatrixCSR::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "------ CSR ------" << std::endl;
  os << "-----------------" << std::endl;
  os << "n_row : " << this->n_row << std::endl;
  os << "n_col : " << this->n_col << std::endl;
  os << "n_values : " << this->values.size() << std::endl;
  tbsla::utils::vector::streamvector<double>(os, "values", this->values);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "rowptr", this->rowptr);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "colidx", this->colidx);
  os << std::endl;
  os << "-----------------" << std::endl;
  return os;
}

std::ostream& tbsla::cpp::MatrixCSR::print_as_dense(std::ostream& os) {
  std::vector<double> d(this->n_row * this->n_col, 0);
  if(this->values.size() != 0) {
    int incr = 0;
    for (int i = 0; i < this->rowptr.size() - 1; i++) {
      for (int j = this->rowptr[i] - this->rowptr.front(); j < this->rowptr[i + 1] - this->rowptr.front(); j++) {
         if(incr < this->values.size())
           d[i * this->n_col + this->colidx[j]] += this->values[j];
         else
           break;
      }
    }
  }
  tbsla::utils::vector::print_dense_matrix(this->n_row, this->n_col, d, os);
  return os;
}

std::vector<double> tbsla::cpp::MatrixCSR::spmv(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r (this->n_row, 0);
  if (this->values.size() == 0)
    return r;
  for (int i = 0; i < this->rowptr.size() - 1; i++) {
    for (int j = this->rowptr[i] - this->rowptr.front(); j < this->rowptr[i + 1] - this->rowptr.front(); j++) {
       r[i + this->f_row] += this->values[j] * v[this->colidx[j]];
    }
  }
  return r;
}

std::vector<double> tbsla::cpp::MatrixCSR::a_axpx_(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r = this->spmv(v, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  r = this->spmv(r, vect_incr);
  return r;
}

std::ostream & tbsla::cpp::MatrixCSR::print_infos(std::ostream &os) {
  os << "-----------------" << std::endl;
  os << "------ CSR ------" << std::endl;
  os << "--- general   ---" << std::endl;
  os << "n_row : " << n_row << std::endl;
  os << "n_col : " << n_col << std::endl;
  os << "--- capacity  ---" << std::endl;
  os << "values : " << values.capacity() << std::endl;
  os << "rowptr : " << rowptr.capacity() << std::endl;
  os << "colidx : " << colidx.capacity() << std::endl;
  os << "--- size      ---" << std::endl;
  os << "values : " << values.size() << std::endl;
  os << "rowptr : " << rowptr.size() << std::endl;
  os << "colidx : " << colidx.size() << std::endl;
  os << "-----------------" << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixCSR::print_stats(std::ostream &os) {
  int s = 0, u = 0, d = 0;
  for (int i = 0; i < this->rowptr.size() - 1; i++) {
    for (int j = this->rowptr[i] - this->rowptr.front(); j < this->rowptr[i + 1] - this->rowptr.front(); j++) {
      if(i < this->colidx[j]) {
        s++;
      } else if(i > this->colidx[j]) {
        u++;
      } else {
        d++;
      }
    }
  }
  os << "upper values : " << u << std::endl;
  os << "lower values : " << s << std::endl;
  os << "diag  values : " << d << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixCSR::write(std::ostream &os) {
  os.write(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  os.write(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));

  size_t size_v = this->values.size();
  os.write(reinterpret_cast<char*>(&size_v), sizeof(size_v));
  os.write(reinterpret_cast<char*>(this->values.data()), this->values.size() * sizeof(double));

  size_t size_r = this->rowptr.size();
  os.write(reinterpret_cast<char*>(&size_r), sizeof(size_r));
  os.write(reinterpret_cast<char*>(this->rowptr.data()), this->rowptr.size() * sizeof(int));

  size_t size_c = this->colidx.size();
  os.write(reinterpret_cast<char*>(&size_c), sizeof(size_c));
  os.write(reinterpret_cast<char*>(this->colidx.data()), this->colidx.size() * sizeof(int));
  return os;
}

std::istream & tbsla::cpp::MatrixCSR::read(std::istream &is, std::size_t pos, std::size_t n) {
  is.read(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  is.read(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));

  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->values.resize(size);
  is.read(reinterpret_cast<char*>(this->values.data()), size * sizeof(double));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->rowptr.resize(size);
  is.read(reinterpret_cast<char*>(this->rowptr.data()), size * sizeof(int));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->colidx.resize(size);
  is.read(reinterpret_cast<char*>(this->colidx.data()), size * sizeof(int));
  return is;
}

void tbsla::cpp::MatrixCSR::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;

  this->values.clear();
  this->colidx.clear();
  this->rowptr.clear();

  ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  f_col = tbsla::utils::range::pflv(n_col, pc, NC);

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

  this->values.reserve(nv);
  this->colidx.reserve(nv);
  this->rowptr.reserve(ln_row + 1);

  size_t incr = 0;
  this->rowptr.push_back(incr);
  for(int i = f_row; i < f_row + ln_row; i++) {
    int ii, jj;
    jj = i - cdiag;
    ii = i;
    if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
      this->colidx.push_back(jj);
      this->values.push_back(1);
      incr++;
    }
    if(cdiag != 0) {
      jj = i + cdiag;
      ii = i;
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->colidx.push_back(jj);
        this->values.push_back(1);
        incr++;
      }
    }
    this->rowptr.push_back(incr);
  }
}

void tbsla::cpp::MatrixCSR::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;

  this->values.clear();
  this->colidx.clear();
  this->rowptr.clear();


  ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  f_col = tbsla::utils::range::pflv(n_col, pc, NC);

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
  int incr_save = incr;

  int i;
  for(i = f_row; i < std::min({n_row, n_col, f_row + ln_row}); i++) {
    for(int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      int ii, jj;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->nnz++;
      }
      incr++;
    }
  }

  incr = incr_save;

  if(nv == 0)
    return;

  this->values.reserve(this->nnz);
  this->colidx.reserve(this->nnz);
  this->rowptr.reserve(ln_row + 1);

  int lincr = 0;
  this->rowptr.push_back(lincr);
  for(i = f_row; i < std::min(min_, f_row + ln_row); i++) {
    for(int j = 0; j < std::min(c, n_col); j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      int ii, jj;
      double v;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      v = std::get<2>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->colidx.push_back(jj);
        this->values.push_back(v);
        lincr++;
      }
      incr++;
    }
    this->rowptr.push_back(lincr);
  }
  for(; i < std::min({n_row, n_col, f_row + ln_row}); i++) {
    for(int j = 0; j < std::min(c, n_col) - i + min_ - 1; j++) {
      auto tuple = tbsla::utils::values_generation::cqmat_value(incr, n_row, n_col, c, q, seed_mult);
      int ii, jj;
      double v;
      ii = std::get<0>(tuple);
      jj = std::get<1>(tuple);
      v = std::get<2>(tuple);
      if(ii >= f_row && ii < f_row + ln_row && jj >= f_col && jj < f_col + ln_col) {
        this->colidx.push_back(jj);
        this->values.push_back(v);
        lincr++;
      }
      incr++;
    }
    this->rowptr.push_back(lincr);
  }

  this->values.shrink_to_fit();
  this->colidx.shrink_to_fit();
  this->rowptr.shrink_to_fit();
}
