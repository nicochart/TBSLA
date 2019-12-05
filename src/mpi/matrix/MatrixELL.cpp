#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <vector>
#include <mpi.h>

int tbsla::mpi::MatrixELL::read_bin_mpiio(MPI_Comm comm, std::string filename) {
  return 0;
}

std::vector<double> tbsla::mpi::MatrixELL::spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
  std::vector<double> send = this->spmv(v, this->row_incr + vect_incr);
  std::vector<double> recv(send.size());
  MPI_Allreduce(send.data(), recv.data(), send.size(), MPI_DOUBLE, MPI_SUM, comm);
  return recv;
}

std::vector<double> tbsla::mpi::MatrixELL::a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
  std::vector<double> r = this->spmv(comm, v, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  return this->spmv(comm, r, vect_incr);
}

void tbsla::mpi::MatrixELL::fill_cdiag(MPI_Comm comm, int nr, int nc, int cdiag) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);
  this->row_incr = tbsla::utils::range::pflv(nr, rank, world);
  this->tbsla::cpp::MatrixELL::fill_cdiag(nr, nc, cdiag, rank, world);
}
