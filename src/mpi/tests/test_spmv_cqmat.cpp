#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/mpi/MatrixELL.hpp>

#include <tbsla/cpp/utils/vector.hpp>

#include <mpi.h>

#include <numeric>
#include <iostream>

void test_cqmat(int nr, int nc, int c, double q, unsigned int seed) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << q << ": " << q << "; s : " << seed << "----  r : " << rank << "/" << world << std::endl;

  std::vector<double> v(nc);
  std::iota (std::begin(v), std::end(v), 0);

  tbsla::mpi::MatrixCOO mcoo;
  mcoo.fill_cqmat(MPI_COMM_WORLD, nr, nc, c, q, seed);
  std::vector<double> rcoo = mcoo.spmv(MPI_COMM_WORLD, v);

  tbsla::mpi::MatrixCSR mcsr;
  mcsr.fill_cqmat(MPI_COMM_WORLD, nr, nc, c, q, seed);
  std::vector<double> rcsr = mcsr.spmv(MPI_COMM_WORLD, v);
  if(rcsr != rcoo) {
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << mcoo << std::endl;
        std::cout << mcsr << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0) {
      tbsla::utils::vector::streamvector<double>(std::cout, "v ", v);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rcsr ", rcsr);
      std::cout << std::endl;
      exit(1);
    }
  }

  tbsla::mpi::MatrixELL mell;
  mell.fill_cqmat(MPI_COMM_WORLD, nr, nc, c, q, seed);
  std::vector<double> rell = mell.spmv(MPI_COMM_WORLD, v);
  if(rell != rcoo) {
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << mcoo << std::endl;
        std::cout << mell << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0) {
      tbsla::utils::vector::streamvector<double>(std::cout, "v ", v);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rell ", rell);
      std::cout << std::endl;
      exit(1);
    }
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

  MPI_Init(&argc, &argv);

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

  MPI_Finalize();
}
