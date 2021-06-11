#include <tbsla/cpp/Matrix.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
double* tbsla::cpp::Matrix::a_axpx_(const double* v, int vect_incr) const {
  double* r = this->spmv(v, vect_incr);
  for (int i = 0; i < this->ln_col; i++) {
    r[i] += v[i];
  }
  r = this->spmv(r, vect_incr);
  return r;
}

