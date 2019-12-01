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
  res = tbsla::utils::vector::test_vres_cdiag(nr, nc, c, r, false);
  std::cout << "return : " << res << std::endl;
  if (res) {
    tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "r", r);
    std::cout << std::endl;
    print(m);
    res = tbsla::utils::vector::test_vres_cdiag(nr, nc, c, r, true);
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

  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(10, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(5, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(10, 5, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(30, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(20, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(30, 20, 2 * i);
  }

  std::cout << "=== finished without error === " << std::endl;
}
