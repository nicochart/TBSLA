#include <tbsla/cpp/Matrix.hpp>
#include <algorithm>
#include <numeric>
#include <vector>

std::vector<double> tbsla::cpp::Matrix::a_axpx_(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r = this->spmv(v, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  r = this->spmv(r, vect_incr);
  return r;
}

