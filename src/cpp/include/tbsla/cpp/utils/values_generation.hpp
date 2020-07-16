#ifndef TBSLA_CPP_VALUES_GENERATION
#define TBSLA_CPP_VALUES_GENERATION

#include <tuple>
#include <vector>

namespace tbsla { namespace utils { namespace values_generation {
  std::tuple<int, int, double, int> cdiag_value(int i, int nv, int nr, int nc, int cdiag);
  std::tuple<int, int, double, int> cqmat_value(int i, int nr, int nc, int c, double q, unsigned int seed_mult);
  std::vector<double> cqmat_sum_columns(int nr, int nc, int c_, double q, unsigned int seed_mult);
}}}

#endif
