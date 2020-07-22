#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/mpi/MatrixSCOO.hpp>
#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/mpi/MatrixDENSE.hpp>

#include <tbsla/cpp/utils/vector.hpp>

#include <mpi.h>

#include <numeric>
#include <iostream>

void test_cqmat(int nr, int nc, int c, double q, unsigned int seed, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<double> v(nc);
  std::iota (std::begin(v), std::end(v), 0);

  tbsla::mpi::MatrixCOO mcoo;
  mcoo.fill_cqmat_stochastic(nr, nc, c, q, seed, pr, pc, NR, NC);
  std::vector<double> rcoo = mcoo.spmv(MPI_COMM_WORLD, v);

  tbsla::mpi::MatrixSCOO mscoo;
  mscoo.fill_cqmat_stochastic(nr, nc, c, q, seed, pr, pc, NR, NC);
  std::vector<double> vl(mscoo.get_ln_col());
  std::iota (std::begin(vl), std::end(vl), mscoo.get_f_col());
  std::vector<double> rscoo = mscoo.spmv(MPI_COMM_WORLD, vl);
  if(tbsla::utils::vector::compare_vectors(rscoo, rcoo)) {
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << mcoo << std::endl;
        std::cout << mscoo << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0) {
      tbsla::utils::vector::streamvector<double>(std::cout, "v ", v);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "vl ", vl);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rscoo ", rscoo);
      std::cout << std::endl;
      exit(1);
    }
  }

  tbsla::mpi::MatrixCSR mcsr;
  mcsr.fill_cqmat_stochastic(nr, nc, c, q, seed, pr, pc, NR, NC);
  std::vector<double> rcsr = mcsr.spmv(MPI_COMM_WORLD, vl);
  if(tbsla::utils::vector::compare_vectors(rcsr, rcoo)) {
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
      tbsla::utils::vector::streamvector<double>(std::cout, "vl ", vl);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rcsr ", rcsr);
      std::cout << std::endl;
      exit(1);
    }
  }

  tbsla::mpi::MatrixELL mell;
  mell.fill_cqmat_stochastic(nr, nc, c, q, seed, pr, pc, NR, NC);
  std::vector<double> rell = mell.spmv(MPI_COMM_WORLD, vl);
  if(tbsla::utils::vector::compare_vectors(rell, rcoo)) {
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
      tbsla::utils::vector::streamvector<double>(std::cout, "vl ", vl);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rell ", rell);
      std::cout << std::endl;
      exit(1);
    }
  }

  tbsla::mpi::MatrixDENSE mdense;
  mdense.fill_cqmat_stochastic(nr, nc, c, q, seed, pr, pc, NR, NC);
  std::vector<double> rdense = mdense.spmv(MPI_COMM_WORLD, vl);
  if(tbsla::utils::vector::compare_vectors(rdense, rcoo)) {
    for(int i = 0; i < world; i++) {
      MPI_Barrier(MPI_COMM_WORLD);
      if(i == rank) {
        std::cout << mcoo << std::endl;
        std::cout << mdense << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0) {
      tbsla::utils::vector::streamvector<double>(std::cout, "v ", v);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "vl ", vl);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
      std::cout << std::endl;
      tbsla::utils::vector::streamvector<double>(std::cout, "rdense ", rdense);
      std::cout << std::endl;
      exit(1);
    }
  }
}

void test_mat(int nr, int nc, int c) {
  for(double s = 0; s < 4; s++) {
    for(double q = 0; q <= 1; q += 0.1) {
      int world, rank;
      MPI_Comm_size(MPI_COMM_WORLD, &world);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0)
        std::cout << "--- row ---- nr : " << nr << "; nc : " << nc << "; c : " << q << ": " << q << "; s : " << s << "----" << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      test_cqmat(nr, nc, c, q, s, rank, 0, world, 1);
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
        std::cout << "--- col ---- nr : " << nr << "; nc : " << nc << "; c : " << q << ": " << q << "; s : " << s << "----" << std::endl;
      MPI_Barrier(MPI_COMM_WORLD);
      test_cqmat(nr, nc, c, q, s, 0, rank, 1, world);
      MPI_Barrier(MPI_COMM_WORLD);
      if(world % 2 == 0 && world / 2 > 1) {
        if (rank == 0)
          std::cout << "--- mix ---- nr : " << nr << "; nc : " << nc << "; c : " << q << ": " << q << "; s : " << s << "----" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        test_cqmat(nr, nc, c, q, s, rank / 2, rank % 2, world / 2, 2);
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int t = 0;
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
