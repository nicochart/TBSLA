#include <tbsla/cpp/Matrix.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
double* tbsla::cpp::Matrix::a_axpx_(const double* v, int vect_incr) const {
  double* r = this->spmv(v, vect_incr);
  for (int i = 0; i < this->ln_col; i++) {
    r[i] += v[i];
  }
  double* r2 = this->spmv(r, vect_incr);
  delete[] r;
  return r2;
}

void tbsla::cpp::Matrix::AAxpAx(double* r, double* v, int vect_incr) const {
  this->Ax(r, v, vect_incr);
  for (int i = 0; i < this->ln_col; i++) {
    r[i] += v[i];
  }
  this->Ax(r, v, vect_incr);
}
