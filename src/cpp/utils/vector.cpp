#include <tbsla/cpp/utils/vector.hpp>

int check(int i, double v, double exp, int return_value, bool debug) {
  if(debug)
    std::cout << "i = " << i << " r[i] = " << v << " exp " << exp << std::endl;
  if(v != exp) {
    return return_value;
  }
  return 0;
}

int tbsla::utils::vector::test_vres_cdiag(int nr, int nc, int c, std::vector<double> r, bool debug) {
  int i;
  for(i = 0; i < std::min(c, nr); i++) {
    int rv = check(i, r[i], i < nc - c ? i + c : 0, 10, debug);
    if(rv) return rv;
  }
  for(; i < std::min(nr, nc - c); i++) {
    int rv = check(i, r[i], c == 0 ? i : 2 * i, 12, debug);
    if(rv) return rv;
  }
  for(; i < nr; i++) {
    int rv = check(i, r[i], i < nc + c ? i - c : 0, 14, debug);
    if(rv) return rv;
  }
  if(i < nr) {
    return 222;
  }
  return 0;
}

