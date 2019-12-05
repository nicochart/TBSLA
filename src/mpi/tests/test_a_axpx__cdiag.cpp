#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/mpi/MatrixELL.hpp>

#include <tbsla/cpp/utils/vector.hpp>

#include <mpi.h>

#include <numeric>
#include <iostream>


void test_matrix(tbsla::mpi::Matrix & m, int nr, int nc, int cdiag) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << cdiag << " ----  r : " << rank << "/" << world << std::endl;

  std::vector<double> v(nc);
  std::iota (std::begin(v), std::end(v), 0);
  std::vector<double> r = m.a_axpx_(MPI_COMM_WORLD, v);
  int res = tbsla::utils::vector::test_a_axpx__cdiag(nr, nc, cdiag, v, r, false);
  int res0;
  MPI_Allreduce(&res, &res0, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(rank == 0) {
    std::cout << "return : " << res0 << std::endl;
  }
  if(res0) {
    int res;
    std::vector<double> r = m.spmv(MPI_COMM_WORLD, v);
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        m.print_infos(std::cout);
        std::cout << m << std::endl;
        res = tbsla::utils::vector::test_a_axpx__cdiag(nr, nc, cdiag, v, r, true);
        tbsla::utils::vector::streamvector<double>(std::cout, "r ", r);
        std::cout << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0) {
      tbsla::cpp::MatrixCOO ml;
      ml.fill_cdiag(nr, nc, cdiag);
      ml.print_infos(std::cout);
      exit(res);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void test_cdiag(int nr, int nc, int cdiag) {
  tbsla::mpi::MatrixCOO mcoo;
  mcoo.fill_cdiag(MPI_COMM_WORLD, nr, nc, cdiag);
  test_matrix(mcoo, nr, nc, cdiag);

  tbsla::mpi::MatrixCSR mcsr;
  mcsr.fill_cdiag(MPI_COMM_WORLD, nr, nc, cdiag);
  test_matrix(mcsr, nr, nc, cdiag);

  tbsla::mpi::MatrixELL mell;
  mell.fill_cdiag(MPI_COMM_WORLD, nr, nc, cdiag);
  test_matrix(mell, nr, nc, cdiag);
}

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(10, 10, i);
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    test_cdiag(30, 30, 2 * i);
  }
  std::cout << "=== finished without error === " << std::endl;

  MPI_Finalize();
}
