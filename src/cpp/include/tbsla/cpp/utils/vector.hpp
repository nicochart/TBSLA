#ifndef TBSLA_CPP_UTILS
#define TBSLA_CPP_UTILS

#include <vector>
#include <iostream>

namespace tbsla { namespace utils { namespace vector {

template <class myType>
void streamvector(std::ostream &os, const std::string name, const std::vector<myType> & v) {
  os << name << " : ";
  auto it = v.cbegin();
  for (; it != v.cend() - 1; it++) {
    os << *it << ", ";
  }
  if (it != v.cend()) {
    os << *it;
  }
}

}}}

#endif
