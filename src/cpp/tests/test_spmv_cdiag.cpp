#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/cpp/utils/vector.hpp>

#include <iostream>

void print(tbsla::cpp::Matrix & m) {
  m.print_infos(std::cout);
  std::cout << "--------" << std::endl;
  std::cout << m << std::endl;
  std::cout << "--------" << std::endl;
}

int test_vres_cdiag(int nr, int nc, int c, std::vector<double> r, bool debug) {
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

void test_matrix(tbsla::cpp::Matrix & m, int c) {
  int nc, nr;
  nc = m.get_n_col();
  nr = m.get_n_row();
  std::vector<double> v(nc);
  for(int i = 0; i < nc; i++) {
    v[i] = i;
  }
  std::vector<double> r = m.spmv(v);
  int res;
  res = test_vres_cdiag(nr, nc, c, r, false);
  std::cout << "return : " << res << std::endl;
  if (res) {
    tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "r", r);
    std::cout << std::endl;
    print(m);
    res = test_vres_cdiag(nr, nc, c, r, true);
    exit(res);
  }
}

void test_cdiag(int nr, int nc, int c) {
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << " ----  " << std::endl;
  tbsla::cpp::MatrixCOO mcoo;
  mcoo.fill_cdiag(nr, nc, c);
  test_matrix(mcoo, c);
  tbsla::cpp::MatrixCSR mcsr;
  mcsr = mcoo.toCSR();
  test_matrix(mcsr, c);
}

int main(int argc, char** argv) {

  for(int i = 0; i <= 10; i++) {
    std::cout << "=== test " << i << " ===" << std::endl;
    test_cdiag(10, 10, i);
    test_cdiag(5, 10, i);
    test_cdiag(10, 5, i);
  }

  std::cout << "=== finished without error === " << std::endl;
}
