#include <tbsla/cpp/utils/cdiag.hpp>

std::tuple<int, int, double, int> tbsla::utils::cdiag::cdiag_value(int i, int nv, int nr, int nc, int cdiag) {
  if(cdiag == 0) {
    return std::make_tuple(i, i, 1, 10);
  }
  if(i < std::max(std::min(nc - cdiag, cdiag), 0)) {
    return std::make_tuple(i, i + cdiag, 1, 30);
  } else if (i < std::max(cdiag + 2 * (nc - 2 * cdiag), 0)) {
    int it = (i - cdiag) / 2 + cdiag;
    if(i % 2 == 0) {
      return std::make_tuple(it, it - cdiag, 1, 31);
    } else {
      return std::make_tuple(it, it + cdiag, 1, 32);
    }
  } else {
    int it = i - (nc - 2 * cdiag);
    if(cdiag > nc) {
      it -= cdiag - nc;
    }
    return std::make_tuple(it, it - cdiag, 1, 33);
  }
}

