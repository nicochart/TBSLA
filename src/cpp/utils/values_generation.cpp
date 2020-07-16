#include <tbsla/cpp/utils/values_generation.hpp>
#include <cstdlib>

std::tuple<int, int, double, int> tbsla::utils::values_generation::cdiag_value(int i, int nv, int nr, int nc, int cdiag) {
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

std::tuple<int, int, double, int> tbsla::utils::values_generation::cqmat_value(int i, int nr, int nc, int c_, double q, unsigned int seed_mult) {
  unsigned int seedp = i;
  if(seed_mult > 0) {
    seedp = seed_mult * i;
  }
  int c = std::min(nc, c_);
  if(i < c * std::min(nc - c + 1, nr)) {
    return std::make_tuple(i / c, ((double)rand_r(&seedp) / RAND_MAX) < q ? rand_r(&seedp) % nc : i / c + i % c, 1, 30);
  } else {
    int n_full_rows = std::min(nc - c + 1, nr);
    int it = i - c * std::min(nc - c + 1, nr);
    int curr_row = 1;
    while(curr_row < nr - n_full_rows && it >= c - curr_row) {
      it -= (c - curr_row);
      curr_row++;
    }
    curr_row += n_full_rows - 1;
    return std::make_tuple(curr_row, (double)(rand_r(&seedp) / RAND_MAX) < q ? rand_r(&seedp) % nc : curr_row + it, 1, 31);
  }
}

std::vector<double> tbsla::utils::values_generation::cqmat_sum_columns(int nr, int nc, int c, double q, unsigned int seed_mult) {
  int gnv = 0;
  for(int i = 0; i < std::min(nc - std::min(c, nc) + 1, nr); i++) {
    gnv += std::min(c, nc);
  }
  for(int i = 0; i < std::min(nr, nc) - std::min(nc - std::min(c, nc) + 1, nr); i++) {
    gnv += std::min(c, nc) - i - 1;
  }

  std::vector<double> sum(nc, 0);
  for(int i = 0; i < gnv; i++) {
    auto tuple = tbsla::utils::values_generation::cqmat_value(i, nr, nc, c, q, seed_mult);
    sum[std::get<1>(tuple)] += std::get<2>(tuple);
  }
  return sum;
}
