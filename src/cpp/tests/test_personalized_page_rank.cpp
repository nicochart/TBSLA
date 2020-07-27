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

void test_personalized_page_rank(int nr, int nc, int c, double q, double s, double beta, double epsilon, int max_iterations, std::vector<int> personalized_nodes) {
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << "; q : " << q << "; s : " << s << " ----  " << std::endl;
  std::vector<double> v(nc);
  std::iota (std::begin(v), std::end(v), 0);

  tbsla::cpp::MatrixCOO mcoo;
  mcoo.fill_cqmat_stochastic(nr, nc, c, q, s);
  std::vector<double> rcoo = mcoo.personalized_page_rank(beta, epsilon, max_iterations, personalized_nodes);

  tbsla::cpp::MatrixSCOO mscoo;
  mscoo.fill_cqmat_stochastic(nr, nc, c, q, s);
  std::vector<double> rscoo = mscoo.personalized_page_rank(beta, epsilon, max_iterations, personalized_nodes);
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

  tbsla::cpp::MatrixCSR mcsr;
  mcsr = mcoo.toCSR();
  std::vector<double> rcsr = mcsr.personalized_page_rank(beta, epsilon, max_iterations, personalized_nodes);
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
  std::vector<double> rcsr2 = mcsr2.personalized_page_rank(beta, epsilon, max_iterations, personalized_nodes);
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
  std::vector<double> rell = mell.personalized_page_rank(beta, epsilon, max_iterations, personalized_nodes);
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
  std::vector<double> rdense = mdense.personalized_page_rank(beta, epsilon, max_iterations, personalized_nodes);
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

void test_mat(int nr, int nc, int c, double beta, double epsilon, int max_iterations, std::vector<int> personalized_nodes) {
  for(double s = 0; s < 4; s++) {
    for(double q = 0; q <= 1; q += 0.1) {
      test_personalized_page_rank(nr, nc, c, q, s, beta, epsilon, max_iterations, personalized_nodes);
    }
  }
}

int main(int argc, char** argv) {
  double epsilon = 0.01;
  double beta = 0.85;
  int max_iterations = 10;
  std::vector<int> personalized_nodes{1,3};
  test_mat(10, 10, 12, beta, epsilon, max_iterations, personalized_nodes);
  test_mat(10, 10, 3, beta, epsilon, max_iterations, personalized_nodes);

  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_mat(30, 30, 2 * i, beta, epsilon, max_iterations, personalized_nodes);
  }
  std::cout << "=== finished without error === " << std::endl;
}
