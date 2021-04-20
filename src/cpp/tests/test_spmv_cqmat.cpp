#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixSCOO.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/cpp/Matrix.hpp>
#include <tbsla/cpp/utils/vector.hpp>

#include <iostream>
#include <numeric>

void print(tbsla::cpp::Matrix & m) {
  m.print_infos(std::cout);
  std::cout << "--------" << std::endl;
  std::cout << m << std::endl;
  std::cout << "--------" << std::endl;
}

void test_cqmat(int nr, int nc, int c, double q, double s) {
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << "; q : " << q << "; s : " << s << " ----  " << std::endl;
  std::vector<double> v(nc);
  std::iota (std::begin(v), std::end(v), 0);

  tbsla::cpp::MatrixCOO mcoo;
  mcoo.fill_cqmat(nr, nc, c, q, s);
  std::vector<double> rcoo = mcoo.spmv(v);

  tbsla::cpp::MatrixSCOO mscoo(mcoo);
  std::vector<double> rscoo = mscoo.spmv(v);
  if(rcoo != rscoo) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mscoo);
    tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rscoo", rscoo);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixSCOO mscoo2;
  mscoo2.fill_cqmat(nr, nc, c, q, s);
  std::vector<double> rscoo2 = mscoo2.spmv(v);
  if(rcoo != rscoo2) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mscoo);
    tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rscoo2", rscoo2);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixCSR mcsr(mcoo);
  std::vector<double> rcsr = mcsr.spmv(v);
  if(rcoo != rcsr) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mcsr);
    tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcsr", rcsr);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixCSR mcsr2;
  mcsr2.fill_cqmat(nr, nc, c, q, s);
  std::vector<double> rcsr2 = mcsr2.spmv(v);
  if(rcoo != rcsr2) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mcsr2);
    tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcsr2", rcsr2);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixELL mell(mcoo);
  std::vector<double> rell = mell.spmv(v);
  if(rcoo != rell) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mell);
    tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rell", rell);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixELL mell2;
  mell2.fill_cqmat(nr, nc, c, q, s);
  std::vector<double> rell2 = mell2.spmv(v);
  if(rcoo != rell2) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mell);
    tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rell2", rell2);
    std::cout << std::endl;
    exit(1);
  }

  tbsla::cpp::MatrixDENSE mdense;
  mdense.fill_cqmat(nr, nc, c, q, s);
  std::vector<double> rdense = mdense.spmv(v);
  if(rcoo != rdense) {
    print(mcoo);
    mcoo.print_as_dense(std::cout);
    print(mdense);
    tbsla::utils::vector::streamvector<double>(std::cout, "v", v);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rdense", rdense);
    std::cout << std::endl;
    exit(1);
  }
}

void test_mat(int nr, int nc, int c) {
  for(double s = 0; s < 4; s++) {
    for(double q = 0; q <= 1; q += 0.1) {
      test_cqmat(nr, nc, c, q, s);
    }
  }
}

int main(int argc, char** argv) {
  test_mat(10, 5, 8);
  test_mat(5, 10, 7);
  test_mat(10, 10, 12);
  test_mat(10, 10, 3);
  test_mat(10, 5, 3);
  test_mat(10, 5, 6);

  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(10, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(5, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(10, 5, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(30, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(20, 30, 2 * i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(30, 20, 2 * i);
  }

  std::cout << "=== finished without error === " << std::endl;
}
