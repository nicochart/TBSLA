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

void test_page_rank(int nr, int nc, int c, double q, double s, double beta, double epsilon, int max_iterations) {
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << "; q : " << q << "; s : " << s << " ----  " << std::endl;
  std::vector<double> v(nc);
  int nb_iterations_done;

  std::iota (std::begin(v), std::end(v), 0);

  tbsla::cpp::MatrixCOO mcoo;
  mcoo.fill_cqmat_stochastic(nr, nc, c, q, s);
  std::vector<double> rcoo = mcoo.page_rank(beta, epsilon, max_iterations, nb_iterations_done);

  tbsla::cpp::MatrixSCOO mscoo;
  mscoo.fill_cqmat_stochastic(nr, nc, c, q, s);
  std::vector<double> rscoo = mscoo.page_rank(beta, epsilon, max_iterations, nb_iterations_done);
  if(tbsla::utils::vector::compare_vectors(rcoo, rscoo)) {
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

  tbsla::cpp::MatrixCSR mcsr(mcoo);
  std::vector<double> rcsr = mcsr.page_rank(beta, epsilon, max_iterations, nb_iterations_done);
  if(tbsla::utils::vector::compare_vectors(rcoo, rcsr)) {
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
  mcsr2.fill_cqmat_stochastic(nr, nc, c, q, s);
  std::vector<double> rcsr2 = mcsr2.page_rank(beta, epsilon, max_iterations, nb_iterations_done);
  if(tbsla::utils::vector::compare_vectors(rcoo, rcsr2)) {
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

  tbsla::cpp::MatrixELL mell;
  mell.fill_cqmat_stochastic(nr, nc, c, q, s);
  std::vector<double> rell = mell.page_rank(beta, epsilon, max_iterations, nb_iterations_done);
  if(tbsla::utils::vector::compare_vectors(rcoo, rell)) {
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

  tbsla::cpp::MatrixDENSE mdense;
  mdense.fill_cqmat_stochastic(nr, nc, c, q, s);
  std::vector<double> rdense = mdense.page_rank(beta, epsilon, max_iterations, nb_iterations_done);
  if(tbsla::utils::vector::compare_vectors(rcoo, rdense)) {
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

void test_mat(int nr, int nc, int c, double beta, double epsilon, int max_iterations) {
  for(double s = 0; s < 4; s++) {
    for(double q = 0; q <= 1; q += 0.1) {
      test_page_rank(nr, nc, c, q, s, beta, epsilon, max_iterations);
    }
  }
}

int main(int argc, char** argv) {
  double epsilon = 0.01;
  double beta = 0.85;
  int max_iterations = 10;

  test_mat(10, 10, 12, beta, epsilon, max_iterations);
  test_mat(10, 10, 3, beta, epsilon, max_iterations);

  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(30, 30, 2 * i, beta, epsilon, max_iterations);
  }
  std::cout << "=== finished without error === " << std::endl;
}
