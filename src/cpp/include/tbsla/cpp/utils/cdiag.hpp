#ifndef TBSLA_CPP_CDIAG
#define TBSLA_CPP_CDIAG

#include <tuple>

namespace tbsla { namespace utils { namespace cdiag {
  std::tuple<int, int, double, int> cdiag_value(int i, int nv, int nr, int nc, int cdiag);
  std::tuple<int, int, double, int> cqmat_value(int i, int nr, int nc, int c, double q, unsigned int seed_mult);
}}}

#endif
