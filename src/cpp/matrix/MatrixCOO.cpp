#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

MatrixCOO::MatrixCOO(int n_row, int n_col, std::vector<double> & values, std::vector<int> & row,  std::vector<int> & col) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->values = values;
  this->row = row;
  this->col = col;
}

MatrixCOO::MatrixCOO(int n_row, int n_col, int n_values) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->values.reserve(n_values);
  this->row.reserve(n_values);
  this->col.reserve(n_values);
}

MatrixCOO::MatrixCOO(int n_row, int n_col) {
  this->n_row = n_row;
  this->n_col = n_col;
}

std::ostream & operator<<( std::ostream &os, const MatrixCOO &m) {
  os << "n_row : " << m.n_row << std::endl;
  os << "n_col : " << m.n_col << std::endl;
  os << "n_values : " << m.values.size() << std::endl;
  tbsla::utils::vector::streamvector<double>(os, "val", m.values);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "row", m.row);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "col", m.col);
  return os;
}

std::vector<double> MatrixCOO::spmv(const std::vector<double> &v) {
  std::vector<double> r (this->n_row, 0);
  for (int i = 0; i < this->values.size(); i++) {
     r[this->row[i]] += this->values[i] * v[this->col[i]];
  }
  return r;
}

void MatrixCOO::push_back(int r, int c, double v) {
  this->values.push_back(v);
  this->row.push_back(r);
  this->col.push_back(c);
}

std::ostream & MatrixCOO::print_infos(std::ostream &os) {
  os << "-----------------" << std::endl;
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

std::ostream & MatrixCOO::print_stats(std::ostream &os) {
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

std::ostream & MatrixCOO::write(std::ostream &os) {
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

std::istream & MatrixCOO::read(std::istream &is) {
  is.read(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  is.read(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));

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

MatrixCSR MatrixCOO::toCSR() {
  std::vector<int> p(this->values.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(), [&](unsigned i, unsigned j){ return compare_row(this->row, this->col, i, j); });

  std::vector<int> pr = applyPermutation<int>(p, this->row);
  std::vector<int> pc = applyPermutation<int>(p, this->col);
  std::vector<double> pv = applyPermutation<double>(p, this->values);

  std::vector<int> cr(this->n_row + 1);
  cr[0] = 0;
  size_t incr = 0;
  for(int i = 1; i < cr.size(); i++) {
    while(this->row[cr[i - 1]] == this->row[incr] && incr <= this->row.size())
      incr++;
    cr[i] = incr;
  }

  return MatrixCSR(this->n_row, this->n_col, pv, cr, pc);
}
