#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <tbsla/cpp/utils/cdiag.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col, std::vector<double> & values, std::vector<int> & row,  std::vector<int> & col) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->values = values;
  this->row = row;
  this->col = col;
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
  os.precision(6);
  os << std::fixed;
  for(int i = 0; i < this->row.size(); i++) {
    d[row[i] * this->n_col + col[i]] += this->values[i];
  }
  for(int i = 0; i < this->n_row; i++) {
    for(int j = 0; j < this->n_col; j++) {
      os << std::setw(9) << d[i * this->n_col + j] << "  ";
    }
    os << std::endl;
  }
  return os;
}

std::ostream& tbsla::cpp::MatrixCOO::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "------ COO ------" << std::endl;
  os << "-----------------" << std::endl;
  os << "n_row : " << this->n_row << std::endl;
  os << "n_col : " << this->n_col << std::endl;
  os << "n_values : " << this->values.size() << std::endl;
  tbsla::utils::vector::streamvector<double>(os, "val", this->values);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "row", this->row);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "col", this->col);
  os << std::endl;
  os << "-----------------" << std::endl;
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


  size_t vec_size, depla_general, depla_local;
  depla_general = 2 * sizeof(int);


  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  depla_general += sizeof(size_t);

  int n_read = size / n;
  int mod = size % n;

  if (pos < mod)
    n_read++;

  this->values.resize(n_read);
  if (pos < mod) {
    depla_local = depla_general + pos * n_read * sizeof(double);
  } else {
    depla_local = depla_general + (pos * n_read + mod) * sizeof(double);
  }
  is.seekg(depla_local);
  is.read(reinterpret_cast<char*>(this->values.data()), n_read * sizeof(double));
  depla_general += size * sizeof(double);

  is.seekg(depla_general);
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  depla_general += sizeof(size_t);

  n_read = size / n;
  mod = size % n;

  if (pos < mod)
    n_read++;

  this->row.resize(n_read);
  if (pos < mod) {
    depla_local = depla_general + pos * n_read * sizeof(int);
  } else {
    depla_local = depla_general + (pos * n_read + mod) * sizeof(int);
  }
  is.seekg(depla_local);
  is.read(reinterpret_cast<char*>(this->row.data()), n_read * sizeof(int));
  depla_general += size * sizeof(int);


  is.seekg(depla_general);
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  depla_general += sizeof(size_t);

  n_read = size / n;
  mod = size % n;

  if (pos < mod)
    n_read++;

  this->col.resize(n_read);
  if (pos < mod) {
    depla_local = depla_general + pos * n_read * sizeof(int);
  } else {
    depla_local = depla_general + (pos * n_read + mod) * sizeof(int);
  }
  is.seekg(depla_local);
  is.read(reinterpret_cast<char*>(this->col.data()), n_read * sizeof(int));
  return is;
}

bool compare_row(std::vector<int> row, std::vector<int> col, unsigned i, unsigned j) {
  if (row[i] == row[j]) {
    return col[i] < col[j];
  }
  return row[i] < row[j];
}

template <typename T>
std::vector<T> applyPermutation(
    const std::vector<int>& order,
    std::vector<T>& t)
{
    assert(order.size() == t.size());
    std::vector<T> st(t.size());
    for(int i=0; i<t.size(); i++)
    {
        st[i] = t[order[i]];
    }
    return st;
}

tbsla::cpp::MatrixCSR tbsla::cpp::MatrixCOO::toCSR() {
  std::vector<int> p(this->values.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(), [&](unsigned i, unsigned j){ return compare_row(this->row, this->col, i, j); });

  std::vector<int> pr = applyPermutation<int>(p, this->row);
  std::vector<int> pc = applyPermutation<int>(p, this->col);
  std::vector<double> pv = applyPermutation<double>(p, this->values);

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

void tbsla::cpp::MatrixCOO::fill_cdiag(int n_row, int n_col, int cdiag, int rp, int RN) {
  this->n_row = n_row;
  this->n_col = n_col;

  this->values.clear();
  this->col.clear();
  this->row.clear();

  int nv = std::max(std::min(n_row, n_col - cdiag), 0) + std::max(std::min(n_row - cdiag, n_col), 0);
  if(cdiag == 0)
    nv /= 2;
  if(nv == 0)
    return;

  this->values.reserve(nv);
  this->col.reserve(nv);
  this->row.reserve(nv);

  int s = tbsla::utils::range::pflv(nv, rp, RN);
  int n = tbsla::utils::range::lnv(nv, rp, RN);
  for(int i = s; i < s + n; i++) {
    auto tuple = tbsla::utils::cdiag::cdiag_value(i, nv, n_row, n_col, cdiag);
    this->push_back(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
  }
}
