#include <tbsla/cpp/utils/vector.hpp>

#include <algorithm>

int check(int i, double v, double exp, int return_value, bool debug) {
  if(debug)
    std::cout << "i = " << i << " r[i] = " << v << " exp " << exp << std::endl;
  if(v != exp) {
    return return_value;
  }
  return 0;
}

std::vector<double> compute_spmv_res_cdiag(int nr, int nc, int c, std::vector<double> & v) {
  std::vector<double> r(nr);
  int i;
  for(i = 0; i < std::min(c, nr); i++) {
    r[i] = i < nc - c ? v[i + c] : 0;
  }
  for(; i < std::min(nr, nc - c); i++) {
    r[i] = c == 0 ? v[i] : v[i + c] + v[i - c];
  }
  for(; i < nr; i++) {
    r[i] = i < nc + c ? v[i - c] : 0;
  }
  return r;
}

int tbsla::utils::vector::test_spmv_cdiag(int nr, int nc, int c, std::vector<double> & v, std::vector<double> & r, bool debug) {
  std::vector<double> rc = compute_spmv_res_cdiag(nr, nc, c, v);
  int i;
  for(i = 0; i < std::min(c, nr); i++) {
    int rv = check(i, r[i], rc[i], 10, debug);
    if(rv) return rv;
  }
  for(; i < std::min(nr, nc - c); i++) {
    int rv = check(i, r[i], rc[i], 12, debug);
    if(rv) return rv;
  }
  for(; i < nr; i++) {
    int rv = check(i, r[i], rc[i], 14, debug);
    if(rv) return rv;
  }
  if(i < nr) {
    return 222;
  }
  return 0;
}

int tbsla::utils::vector::test_a_axpx__cdiag(int nr, int nc, int c, std::vector<double> & v, std::vector<double> & r, bool debug) {
  if(nr != nc) {
    return 111;
  }
  std::vector<double> rc = compute_spmv_res_cdiag(nr, nc, c, v);
  std::transform (rc.begin(), rc.end(), v.begin(), rc.begin(), std::plus<double>());
  rc = compute_spmv_res_cdiag(nr, nc, c, rc);
  int i;
  for(i = 0; i < std::min(c, nr); i++) {
    int rv = check(i, r[i], rc[i], 10, debug);
    if(rv) return rv;
  }
  for(; i < std::min(nr, nc - c); i++) {
    int rv = check(i, r[i], rc[i], 12, debug);
    if(rv) return rv;
  }
  for(; i < nr; i++) {
    int rv = check(i, r[i], rc[i], 14, debug);
    if(rv) return rv;
  }
  if(i < nr) {
    return 222;
  }
  return 0;
}

