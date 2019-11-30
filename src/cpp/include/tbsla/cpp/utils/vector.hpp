#ifndef TBSLA_CPP_UTILS
#define TBSLA_CPP_UTILS

#include <vector>
#include <iostream>

namespace tbsla { namespace utils { namespace vector {

template <class myType>
void streamvector(std::ostream &os, const std::string name, const std::vector<myType> & v) {
  os << name << " : ";
  auto it = v.cbegin();
  if (it != v.cend() && v.size() != 0) {
    for (; it != v.cend() - 1; it++) {
      os << *it << ", ";
    }
    os << *it;
  }
}

int test_vres_cdiag(int nr, int nc, int c, std::vector<double> r, bool debug);

}}}

#endif
