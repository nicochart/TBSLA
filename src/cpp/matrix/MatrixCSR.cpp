#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <iostream>
#include <vector>
#include <string>

MatrixCSR::MatrixCSR(int n_row, int n_col, std::vector<double> & values, std::vector<int> & rowptr,  std::vector<int> & colidx) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->values = values;
  this->rowptr = rowptr;
  this->colidx = colidx;
}

std::ostream & operator<<( std::ostream &os, const MatrixCSR &m) {
  os << "n_row : " << m.n_row << std::endl;
  os << "n_col : " << m.n_col << std::endl;
  os << "n_values : " << m.values.size() << std::endl;
  tbsla::utils::vector::streamvector<double>(os, "values", m.values);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "rowptr", m.rowptr);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "colidx", m.colidx);
  return os;
}

std::vector<double> MatrixCSR::spmv(const std::vector<double> &v) {
  std::vector<double> r (this->n_row, 0);
  for (int i = 0; i < this->n_row; i++) {
    for (int j = this->rowptr[i]; j < this->rowptr[i + 1]; j++) {
       r[i] += this->values[j] * v[this->colidx[j]];
    }
  }
  return r;
}
