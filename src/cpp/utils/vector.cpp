#include <tbsla/cpp/utils/vector.hpp>

int tbsla::utils::vector::test_vres_cdiag(int nr, int nc, int c, std::vector<double> r, bool debug) {
  int i = 0;
  if( c == 0) {
    for(; i < std::min(nc, nr); i++) {
      if(debug)
        std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << i << std::endl;
      if(r[i] != i) {
        return 10;
      }
    }
    for(; i < nr; i++) {
      if(debug)
        std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << 0 << std::endl;
      if(r[i] != 0) {
        return 11;
      }
    }
  }
  for(; i < std::min(c, nc - c); i++) {
    if(debug)
      std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << i + c << std::endl;
    if(r[i] != i + c) {
      return 20;
    }
  }
  for(; i < std::min(c, nr); i++) {
    if(debug)
      std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << 0 << std::endl;
    if(r[i] != 0) {
      return 21;
    }
  }
  if(nr < nc - c) {
    if(debug)
      std::cout << "case nr < nc - c" << std::endl;
    for(; i < nr; i++) {
      if(debug)
        std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << 2 * i << std::endl;
      if(r[i] != 2 * i) {
        return 30;
      }
    }
  } else if(nr < nc + c) {
    if(debug)
      std::cout << "case nr < nc + c" << std::endl;
    for(; i < std::min(nr, nc) - c; i++) {
      if(debug)
        std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << 2 * i << std::endl;
      if(r[i] != 2 * i) {
        return 40;
      }
    }
    for(; i < std::min(nr, nc); i++) {
      if(debug)
        std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << i - c << std::endl;
      if(r[i] != i - c) {
        return 41;
      }
    }
    for(; i < nr; i++) {
      if(debug)
        std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << i - c << std::endl;
      if(r[i] != i - c) {
        return 42;
      }
    }
  } else {
    if(debug)
      std::cout << "case nr >= nc + c" << std::endl;
    for(; i < std::min(nr, nc - c); i++) {
      if(debug)
        std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << 2 * i << std::endl;
      if(r[i] != 2 * i) {
        return 50;
      }
    }
    for(; i < std::min(nr, nc + c); i++) {
      if(debug)
        std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << i - c << std::endl;
      if(r[i] != i - c) {
        return 51;
      }
    }
    for(; i < nr; i++) {
      if(debug)
        std::cout << "i = " << i << " r[i] = " << r[i] << " exp " << 0 << std::endl;
      if(r[i] != 0) {
        return 52;
      }
    }
  }
  if(i < nr) {
    return 222;
  }
  return 0;
}

